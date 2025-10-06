"""
Main evaluation runner for SSMG experiments
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import sys
sys.path.append('src')

from ssmg.integration import SSMGDialogueAgent, LLMInterface
from ssmg.summarizer import SummaryConfig
from baselines import FullHistoryBaseline, SlidingWindowBaseline, RAGBaseline
from metrics import SSMGEvaluator, EvaluationMetrics
from data.loaders import load_dataset, ConfigLoader
from data.loaders import MultiWOZLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runs SSMG experiments and comparisons"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.evaluator = SSMGEvaluator()

        # Initialize LLM interface
        self.llm = LLMInterface(
            model_name=config['llm']['model_name']
        )

    def run_ssmg_experiment(self, dataset_config: Dict) -> Dict[str, Any]:
        """Run SSMG on MultiWOZ dataset"""
        logger.info("Running SSMG experiment on MultiWOZ")
    
        # Load MultiWOZ data
        multiwoz_loader = MultiWOZLoader(
            data_path=dataset_config.get('data_path', 'data/datasets'),
            split='validation',
            max_dialogues=dataset_config.get('max_dialogues', 100)
        )
    
        eval_data = multiwoz_loader.get_evaluation_data()
    
        # Initialize SSMG agent
        agent = SSMGDialogueAgent(
            llm_interface=self.llm,
            graph_config=self.config['graph'],
            summary_config=SummaryConfig(**self.config['summarizer'])
        )
    
        all_turns = []
        all_responses = []
        all_metrics = []
    
        for dialogue in eval_data['dialogues'][:dataset_config.get('max_dialogues', 50)]:
            dialogue_id = dialogue['dialogue_id']
        
            # Start session for this dialogue
            agent.start_session(f"ssmg_{dialogue_id}")
        
            for turn_data in dialogue['turns']:
                user_input = turn_data['user_input']
            
                try:
                    response, metrics = agent.process_turn(user_input)
                
                    turn_record = {
                        'dialogue_id': dialogue_id,
                        'user_input': user_input,
                        'expected_response': turn_data.get('system_response', ''),
                        'extraction_result': {'dominant_intent': 'inform'},  # Simplified
                        'domains': dialogue['domains']
                    }
                
                    all_turns.append(turn_record)
                    all_responses.append(response)
                    all_metrics.append(metrics)
                
                except Exception as e:
                    logger.error(f"Error processing turn in {dialogue_id}: {e}")
                    continue
        
            agent.end_session()
    
        # Evaluate results
        eval_metrics = self.evaluator.evaluate_session(all_turns, all_responses, all_metrics)
    
        return {
            'method': 'ssmg',
            'eval_metrics': eval_metrics,
            'dataset_info': {
                'name': 'MultiWOZ',
                'dialogues_processed': len(eval_data['dialogues']),
                'total_turns': len(all_turns),
                'domains': eval_data['domains']
            },
            'raw_results': {
                'turns': all_turns,
                'responses': all_responses,
                'metrics': all_metrics
            }
        }


    def run_baseline_experiments(self, test_dialogues: List[Dict]) -> Dict[str, Any]:
        """Run baseline methods on test dialogues"""
        baselines = {
            'full_history': FullHistoryBaseline(self.llm),
            'sliding_window_3': SlidingWindowBaseline(self.llm, window_size=3),
            'sliding_window_5': SlidingWindowBaseline(self.llm, window_size=5),
            'rag_3': RAGBaseline(self.llm, max_retrieved=3)
        }

        results = {}

        for baseline_name, baseline in baselines.items():
            logger.info(f"Running {baseline_name} baseline")

            all_turns = []
            all_responses = []
            all_metrics = []

            for dialogue in test_dialogues:
                dialogue_id = dialogue['name']
                turns = dialogue['turns']

                baseline.reset()

                for i, turn_text in enumerate(turns):
                    try:
                        result = baseline.process_turn(turn_text)

                        turn_data = {
                            'dialogue_id': dialogue_id,
                            'user_input': turn_text,
                            'extraction_result': {'dominant_intent': 'order'}
                        }

                        # Create mock metrics compatible with evaluation
                        mock_metrics = type('MockMetrics', (), {
                            'context_tokens': result.context_tokens,
                            'total_time': result.latency,
                            'turn_id': i
                        })()

                        all_turns.append(turn_data)
                        all_responses.append(result.response)
                        all_metrics.append(mock_metrics)

                    except Exception as e:
                        logger.error(f"Error in {baseline_name}: {e}")
                        continue

            # Evaluate baseline
            eval_metrics = self.evaluator.evaluate_session(all_turns, all_responses, all_metrics)

            results[baseline_name] = {
                'method': baseline_name,
                'eval_metrics': eval_metrics,
                'raw_results': {
                    'turns': all_turns,
                    'responses': all_responses,
                    'metrics': all_metrics
                }
            }

        return results

    def run_ablation_studies(self, test_dialogues: List[Dict]) -> Dict[str, Any]:
        """Run ablation studies on SSMG parameters"""
        logger.info("Running ablation studies")

        ablation_configs = [
            {'name': 'ttl_3', 'graph': {**self.config['graph'], 'max_ttl_turns': 3}},
            {'name': 'ttl_5', 'graph': {**self.config['graph'], 'max_ttl_turns': 5}},
            {'name': 'ttl_10', 'graph': {**self.config['graph'], 'max_ttl_turns': 10}},
            {'name': 'nodes_20', 'graph': {**self.config['graph'], 'max_nodes': 20}},
            {'name': 'nodes_30', 'graph': {**self.config['graph'], 'max_nodes': 30}},
            {'name': 'nodes_100', 'graph': {**self.config['graph'], 'max_nodes': 100}}
        ]

        ablation_results = {}

        for config in ablation_configs:
            logger.info(f"Running ablation: {config['name']}")

            agent = SSMGDialogueAgent(
                llm_interface=self.llm,
                graph_config=config['graph'],
                summary_config=SummaryConfig(**self.config['summarizer'])
            )

            all_turns = []
            all_responses = []
            all_metrics = []

            for dialogue in test_dialogues:
                agent.start_session(f"ablation_{config['name']}_{dialogue['name']}")

                for turn_text in dialogue['turns']:
                    try:
                        response, metrics = agent.process_turn(turn_text)

                        turn_data = {
                            'dialogue_id': dialogue['name'],
                            'user_input': turn_text,
                            'extraction_result': {'dominant_intent': 'order'}
                        }

                        all_turns.append(turn_data)
                        all_responses.append(response)
                        all_metrics.append(metrics)

                    except Exception as e:
                        logger.error(f"Ablation error: {e}")
                        continue

                agent.end_session()

            eval_metrics = self.evaluator.evaluate_session(all_turns, all_responses, all_metrics)
            ablation_results[config['name']] = eval_metrics

        return ablation_results

    def generate_plots(self, results: Dict[str, Any], output_dir: str):
        """Generate evaluation plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Prepare data for plotting
        methods = []
        accuracies = []
        latencies = []
        token_usages = []

        for method, result in results.items():
            if 'eval_metrics' in result:
                metrics = result['eval_metrics']
                methods.append(method)
                accuracies.append(metrics.turn_accuracy)
                latencies.append(metrics.avg_latency)
                token_usages.append(metrics.avg_context_tokens)

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Accuracy comparison
        axes[0, 0].bar(methods, accuracies)
        axes[0, 0].set_title('Turn Accuracy by Method')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Latency comparison
        axes[0, 1].bar(methods, latencies)
        axes[0, 1].set_title('Average Latency by Method')
        axes[0, 1].set_ylabel('Latency (s)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Token usage comparison
        axes[1, 0].bar(methods, token_usages)
        axes[1, 0].set_title('Average Context Tokens by Method')
        axes[1, 0].set_ylabel('Tokens')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Efficiency scatter plot
        axes[1, 1].scatter(token_usages, accuracies)
        for i, method in enumerate(methods):
            axes[1, 1].annotate(method, (token_usages[i], accuracies[i]))
        axes[1, 1].set_xlabel('Context Tokens')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy vs Token Efficiency')

        plt.tight_layout()
        plt.savefig(output_path / 'comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Plots saved to {output_path / 'comparison_plots.png'}")

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file"""

        # Convert evaluation metrics to serializable format
        serializable_results = {}
        for method, result in results.items():
            if 'eval_metrics' in result:
                metrics = result['eval_metrics']
                serializable_results[method] = {
                    'turn_accuracy': metrics.turn_accuracy,
                    'task_success_rate': metrics.task_success_rate,
                    'avg_context_tokens': metrics.avg_context_tokens,
                    'avg_latency': metrics.avg_latency,
                    'reference_resolution_accuracy': metrics.reference_resolution_accuracy,
                    'constraint_adherence': metrics.constraint_adherence,
                    'token_efficiency': metrics.token_efficiency,
                    'latency_efficiency': metrics.latency_efficiency
                }
            else:
                serializable_results[method] = result

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Run SSMG evaluation on MultiWOZ')
    parser.add_argument('--config', default='configs/default_config.json')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--max-dialogues', type=int, default=100, 
                       help='Maximum number of dialogues to evaluate')
    parser.add_argument('--run-baselines', action='store_true')
    parser.add_argument('--run-ablations', action='store_true')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader.load_config(args.config)
    
    # Update config with MultiWOZ settings
    config['evaluation']['max_dialogues'] = args.max_dialogues
    
    # Create output directory  
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize runner
    runner = ExperimentRunner(config)
    
    # Run experiments
    results = {}
    
    # Run SSMG on MultiWOZ
    ssmg_results = runner.run_ssmg_experiment(config['evaluation'])
    results['ssmg'] = ssmg_results
    
    # Run baselines if requested
    if args.run_baselines:
        baseline_results = runner.run_baseline_experiments(config['evaluation'])
        results.update(baseline_results)
    
    # Save and visualize results
    runner.save_results(results, output_dir / 'multiwoz_results.json')
    runner.generate_plots(results, output_dir)
    
    # Print summary
    print("\n🎯 MultiWOZ Evaluation Results:")
    for method, result in results.items():
        if 'eval_metrics' in result:
            metrics = result['eval_metrics']
            print(f"{method}: Accuracy={metrics.turn_accuracy:.3f}, "
                  f"Tokens/Turn={metrics.avg_context_tokens:.1f}, "
                  f"Latency={metrics.avg_latency:.3f}s")