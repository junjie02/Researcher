# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
import re
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


from o2searcher.rewards.rewards_format import format_reward_fn
from o2searcher.rewards.rewards_score import batch_f1_reward_fn, dv_reward_fn, f1_reward_fn

CLOSED_ENDED_SOURCES = ['nq_hotpotqa', 'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']


def _adjust_panality(panality, data_source):
    if data_source in CLOSED_ENDED_SOURCES:
        return 0.5*panality
    else:
        return 0.2*panality


def _config_get(config, key, default):
    if config is None:
        return default
    if hasattr(config, 'get'):
        return config.get(key, default)
    return getattr(config, key, default)


def _extract_last_answer(text):
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not matches:
        return ''
    return matches[-1].strip()


def _target_list(ground_truth):
    target = ground_truth['target']
    if hasattr(target, 'tolist'):
        target = target.tolist()
    if isinstance(target, tuple):
        target = list(target)
    if not isinstance(target, list):
        target = [target]
    return target


def _efficiency_reward(t_c, d, max_turns, eps=1e-6):
    if t_c < 0:
        return 0.0
    if t_c == 0:
        return 0.4 - 0.05 * d

    reward = 0.4 * min(d, t_c) / (t_c + eps)
    if d > t_c:
        reward -= 0.05 * (d - t_c)
        if d >= max_turns:
            reward -= 0.1
    return float(reward)


def _query_list(queries):
    if queries is None:
        return []
    if isinstance(queries, str):
        return [q for q in queries.split('\n') if q.strip()]
    return [q for q in queries if str(q).strip()]


def _base_reward(solution_str, ground_truth, queries, data_source):
    format_score = format_reward_fn(solution_str).reward
    dv_score = dv_reward_fn(_query_list(queries))
    if data_source in CLOSED_ENDED_SOURCES:
        accuracy_score = qa_em.compute_score_em(solution_str=solution_str, ground_truth=ground_truth, queries=queries)
    else:
        accuracy_score = f1_reward_fn(solution_str, _target_list(ground_truth))

    weights = [1.0, 1.0, 0.5]
    base_score = 0.5 * (
        weights[0] * format_score +
        weights[1] * accuracy_score +
        weights[2] * dv_score
    ) / sum(weights)
    return float(base_score), {
        'format_mean': float(format_score),
        'accuracy_mean': float(accuracy_score),
        'dv_mean': float(dv_score),
        'format_weighted_mean': float(0.5 * weights[0] * format_score / sum(weights)),
        'accuracy_weighted_mean': float(0.5 * weights[1] * accuracy_score / sum(weights)),
        'dv_weighted_mean': float(0.5 * weights[2] * dv_score / sum(weights)),
    }


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, agent_config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.agent_config = agent_config

        efficiency_config = _config_get(agent_config, 'efficiency_reward', {})
        self.efficiency_enabled = bool(_config_get(efficiency_config, 'enable', True))
        self.f1_threshold = float(_config_get(efficiency_config, 'f1_threshold', 0.85))
        self.efficiency_weight = float(_config_get(efficiency_config, 'weight', 0.25))
        self.max_turns = int(_config_get(agent_config, 'max_turns', 5))

    def _score_intermediate_answers(self, data_source, ground_truth, candidate_texts):
        answers = [_extract_last_answer(text) for text in candidate_texts]
        if data_source in CLOSED_ENDED_SOURCES:
            golden_answers = _target_list(ground_truth)
            return [1.0 if qa_em.em_check(answer, golden_answers) else 0.0 for answer in answers]

        references = _target_list(ground_truth)
        return batch_f1_reward_fn(candidate_texts, references, threshold=self.f1_threshold)

    def _compute_probe_rewards(self, data_source, ground_truth, candidate_texts, candidate_depths, actual_search_depth):
        if not candidate_texts:
            return -1, 0.0, 0.0, 0.0

        success_scores = self._score_intermediate_answers(data_source, ground_truth, candidate_texts)
        success_flags = []
        for score in success_scores:
            if data_source in CLOSED_ENDED_SOURCES:
                success_flags.append(score == 1.0)
            else:
                success_flags.append(score >= self.f1_threshold)

        ordered = sorted(
            zip(candidate_depths, success_flags),
            key=lambda item: item[0],
        )
        t_c = -1
        for depth, success in ordered:
            if success:
                t_c = int(depth)
                break

        raw_efficiency = _efficiency_reward(t_c, actual_search_depth, self.max_turns)
        over_search = max(0, actual_search_depth - t_c) if t_c >= 0 else 0
        success_ratio = 1.0 if t_c >= 0 else 0.0
        return t_c, raw_efficiency, over_search, success_ratio

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            queries = data_item.meta_info['queries'][i]
            panality = data_item.meta_info['penalties'][i]
            base_score, base_metrics = _base_reward(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                queries=queries,
                data_source=data_source,
            )

            actual_search_depth = int(data_item.meta_info.get('valid_search_stats', [0] * len(data))[i])
            intermediate_answers = data_item.meta_info.get('intermediate_answers', [[] for _ in range(len(data))])[i]
            intermediate_depths = data_item.meta_info.get('intermediate_depths', [[] for _ in range(len(data))])[i]
            candidate_texts = list(intermediate_answers)
            candidate_depths = [int(depth) for depth in intermediate_depths]
            candidate_texts.append(sequences_str)
            candidate_depths.append(actual_search_depth)

            t_c, raw_efficiency, over_search, tc_success = self._compute_probe_rewards(
                data_source=data_source,
                ground_truth=ground_truth,
                candidate_texts=candidate_texts,
                candidate_depths=candidate_depths,
                actual_search_depth=actual_search_depth,
            )

            efficiency_component = self.efficiency_weight * raw_efficiency if self.efficiency_enabled else 0.0
            score = base_score + efficiency_component

            metrics = {
                'base_mean': float(base_score),
                **base_metrics,
                'raw_efficiency_mean': float(raw_efficiency),
                'efficiency_mean': float(efficiency_component),
                'final_mean': float(score),
                'efficiency_weight': float(self.efficiency_weight),
                'tc_mean': float(t_c),
                'tc_success_ratio': float(tc_success),
                'over_search_mean': float(over_search),
            }
            
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length, metrics

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        reward_metrics = {}
        for i, score, valid_response_length, metrics in results:
            reward_index = max(valid_response_length - 1, 0)
            reward_tensor[i, reward_index] = score
            for key, value in metrics.items():
                reward_metrics.setdefault(key, [0.0] * len(data))
                reward_metrics[key][i] = value

        data.meta_info['reward_metrics'] = reward_metrics

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, agent_config=config.agent)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, agent_config=config.agent)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
