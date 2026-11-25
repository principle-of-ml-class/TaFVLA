"""Separate loader worker settings without real data or training."""
from __future__ import annotations

import unittest

import numpy as np
import torch
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from script.train_adapter import _loader_worker_kwargs, validate
from tests.test_pipeline import _sandbox_tensor_transport
from tests.test_refactor import compose_config
from utils.alignment import collect_alignment_embeddings, compute_alignment_metrics
from utils.common import validate_config
from utils.config import TrainingConfig
from utils.logging import METRIC_KEYS


class _EvalSamples(Dataset):
    def __init__(self):
        self.images = np.arange(8*12, dtype=np.float32).reshape(8,1,3,2,2) + 1
        self.force = np.arange(8*6, dtype=np.float32).reshape(8,1,6) + 1
        self.pressure = np.arange(8*144, dtype=np.float32).reshape(8,1,12,12) + 1

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {'tactile':torch.from_numpy(self.images[index].copy()),
                'force':torch.from_numpy(self.force[index].copy()),
                'pressure':torch.from_numpy(self.pressure[index].copy()),
                'metadata':{'sample_id':f'sample-{index}',
                            'frame_indices':torch.tensor([index]),
                            'timestamps':torch.tensor([index/30],dtype=torch.float64)}}


class _EvalModel(torch.nn.Module):
    def forward(self, *, tactile, force, pressure):
        assert not self.training and not torch.is_grad_enabled()
        value = force.mean() + pressure.mean() / 100
        return {**{key:value for key in METRIC_KEYS},
                'z_tactile':tactile[:,0,:,0,0], 'c_force':force[:,0,:3]}


def _eval_loader(dataset, workers, *, alignment=False):
    kwargs = _loader_worker_kwargs(workers, 1)
    if workers:
        # The execution sandbox forbids PyTorch's FD-sharing socket. Test only:
        # transport tensors by value; the production loader is unchanged.
        kwargs.update(multiprocessing_context='spawn', worker_init_fn=_sandbox_tensor_transport,
                      timeout=10)
    generator = torch.Generator().manual_seed(42)
    if alignment:
        return DataLoader(Subset(dataset,[7,2,0,4]),batch_size=2,shuffle=False,
                          generator=generator,**kwargs)
    return DataLoader(dataset,batch_sampler=[[6,1],[7,0],[3,5],[2,4]],
                      generator=generator,**kwargs)


def _shutdown(loader):
    iterator = getattr(loader,'_iterator',None)
    if iterator is not None:
        workers = list(iterator._workers)
        iterator._shutdown_workers()
        return workers
    return []


class EvaluationWorkerTests(unittest.TestCase):
    def test_defaults_keep_four_training_workers_and_zero_eval_workers(self):
        cfg = compose_config()
        validate_config(cfg)
        self.assertEqual(TrainingConfig().eval_num_workers,0)
        self.assertEqual(cfg.training.num_workers,4)
        self.assertEqual(cfg.training.eval_num_workers,0)
        train_kwargs = _loader_worker_kwargs(cfg.training.num_workers,cfg.training.prefetch_factor)
        eval_kwargs = _loader_worker_kwargs(cfg.training.eval_num_workers,cfg.training.prefetch_factor)
        self.assertNotIn('prefetch_factor',eval_kwargs)
        dataset = _EvalSamples()
        # Constructing the training loader does not spawn or run training workers.
        train = DataLoader(dataset,batch_size=256,**train_kwargs)
        validation = _eval_loader(dataset,cfg.training.eval_num_workers)
        alignment = _eval_loader(dataset,cfg.training.eval_num_workers,alignment=True)
        self.assertEqual(train.num_workers,4)
        self.assertTrue(train.persistent_workers)
        self.assertEqual(train.prefetch_factor,1)
        for loader in (validation,alignment):
            self.assertEqual(loader.num_workers,0)
            self.assertFalse(loader.persistent_workers)
            self.assertIsNone(loader.prefetch_factor)
            self.assertGreater(len(list(loader)),0)

    def test_eval_override_two_workers_and_config_validation(self):
        cfg = compose_config('training.eval_num_workers=2')
        validate_config(cfg)
        self.assertEqual(cfg.training.num_workers,4)
        self.assertEqual(cfg.training.eval_num_workers,2)
        kwargs = _loader_worker_kwargs(cfg.training.eval_num_workers,cfg.training.prefetch_factor)
        self.assertEqual(kwargs['num_workers'],2)
        self.assertTrue(kwargs['persistent_workers'])
        self.assertEqual(kwargs['prefetch_factor'],1)
        for text in ('-1','1.5','two','true','null','[1,2]'):
            with self.subTest(override=text):
                try:
                    invalid = compose_config(f'training.eval_num_workers={text}')
                except ConfigCompositionException:
                    continue
                with self.assertRaises((ValueError,TypeError)):
                    validate_config(invalid)
        for value in (-1,1.5,True,'two',None,[1,2]):
            with self.subTest(value=value):
                raw = OmegaConf.to_container(cfg,resolve=True)
                raw['training']['eval_num_workers'] = value
                with self.assertRaisesRegex((ValueError,TypeError),'training.eval_num_workers'):
                    validate_config(OmegaConf.create(raw))

    def test_zero_and_two_workers_preserve_tensors_order_metrics_and_global_rng(self):
        cfg = compose_config('training.eval_num_workers=2')
        dataset = _EvalSamples()
        val0,val2 = _eval_loader(dataset,0),_eval_loader(dataset,cfg.training.eval_num_workers)
        align0,align2 = _eval_loader(dataset,0,alignment=True),_eval_loader(dataset,2,alignment=True)
        rng = torch.get_rng_state().clone()
        model = _EvalModel()
        try:
            for _ in range(2):
                reference,candidate = list(val0),list(val2)
                self.assertEqual([s for batch in candidate for s in batch['metadata']['sample_id']],
                                 [f'sample-{i}' for i in [6,1,7,0,3,5,2,4]])
                for left,right in zip(reference,candidate,strict=True):
                    for key in ('tactile','force','pressure'):
                        self.assertTrue(torch.equal(left[key],right[key]),key)
                    for key in ('frame_indices','timestamps'):
                        self.assertTrue(torch.equal(left['metadata'][key],right['metadata'][key]),key)
            self.assertEqual(validate(model,val0,torch.device('cpu'),False,None),
                             validate(model,val2,torch.device('cpu'),False,None))
            expected_gallery = [f'sample-{i}' for i in [7,2,0,4]]
            for loader in (align0,align2):
                self.assertEqual([s for batch in loader for s in batch['metadata']['sample_id']],
                                 expected_gallery)
            reference = collect_alignment_embeddings(model,align0,torch.device('cpu'))
            candidate = collect_alignment_embeddings(model,align2,torch.device('cpu'))
            for left,right in zip(reference,candidate,strict=True):
                self.assertTrue(torch.equal(left,right))
            self.assertEqual(compute_alignment_metrics(*reference).metrics,
                             compute_alignment_metrics(*candidate).metrics)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        finally:
            for loader in (val2,align2):
                workers = _shutdown(loader)
                self.assertTrue(all(not worker.is_alive() for worker in workers))


if __name__ == '__main__':
    unittest.main()
