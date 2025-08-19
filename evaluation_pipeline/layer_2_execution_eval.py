"""
Layer 2: Workflow Execution Evaluation
Tests the actual execution capability and stability of generated workflows
"""
import json
import os
import time
import asyncio
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .config import EvaluationConfig
from .utils import (
    retry_with_backoff,
    capture_exception_details,
    save_checkpoint,
    load_checkpoint,
    create_batch_iterator,
    merge_results
)


class ExecutionEvaluator:
    """Evaluates workflow execution capability"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.layer_num = 2
    
    def execute_workflow_with_test_inputs(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow with generated test inputs
        
        Args:
            workflow_data: Test data including workflow definition and test inputs
            
        Returns:
            Execution results
        """
        start_time = time.time()
        
        try:
            # Load workflow from Layer 1 results if available
            workflow_json = workflow_data.get('workflow_json')
            if not workflow_json:
                raise ValueError("No workflow JSON found in test data")
            
            # Generate test inputs based on workflow inputs specification
            test_inputs = self._generate_test_inputs(workflow_data)
            
            # Execute workflow with each test input set
            execution_results = []
            for i, test_input in enumerate(test_inputs):
                try:
                    result = self._execute_single_workflow(workflow_json, test_input)
                    execution_results.append({
                        'input_set': i + 1,
                        'success': True,
                        'input': test_input,
                        'output': result,
                        'execution_time': result.get('execution_time', 0)
                    })
                except Exception as e:
                    execution_results.append({
                        'input_set': i + 1,
                        'success': False,
                        'input': test_input,
                        'error': capture_exception_details(e),
                        'execution_time': time.time() - start_time
                    })
            
            # Calculate success rate and statistics
            successful_executions = [r for r in execution_results if r['success']]
            failed_executions = [r for r in execution_results if not r['success']]
            
            # Analyze error types
            error_analysis = self._analyze_errors(failed_executions)
            
            return {
                'test_id': workflow_data.get('test_id'),
                'workflow_name': workflow_data.get('workflow_name'),
                'overall_success': len(successful_executions) > 0,
                'execution_results': execution_results,
                'statistics': {
                    'total_executions': len(execution_results),
                    'successful_executions': len(successful_executions),
                    'failed_executions': len(failed_executions),
                    'success_rate': len(successful_executions) / len(execution_results) if execution_results else 0,
                    'average_execution_time': sum(r.get('execution_time', 0) for r in successful_executions) / len(successful_executions) if successful_executions else 0
                },
                'error_analysis': error_analysis,
                'total_execution_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'test_id': workflow_data.get('test_id'),
                'workflow_name': workflow_data.get('workflow_name'),
                'overall_success': False,
                'error': capture_exception_details(e),
                'total_execution_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_test_inputs(self, workflow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate test inputs based on workflow input specifications
        
        Args:
            workflow_data: Workflow test data
            
        Returns:
            List of test input sets
        """
        workflow_inputs = workflow_data.get('workflow_inputs', [])
        
        # Generate multiple test input sets with different scenarios
        test_input_sets = []
        
        # Basic test case based on the example (RSS content analysis)
        if workflow_data.get('workflow_name') == 'content_analysis':
            test_input_sets = [
                {
                    'article_content': 'This is a sample news article about technology trends and artificial intelligence developments in 2024.',
                    'article_title': 'AI Technology Trends in 2024',
                    'source_category': 'technology'
                },
                {
                    'article_content': 'Breaking news: Stock market shows significant volatility amid economic uncertainty and inflation concerns.',
                    'article_title': 'Market Volatility Continues',
                    'source_category': 'finance'
                },
                {
                    'article_content': 'Local sports team wins championship after an exciting final match that went into overtime.',
                    'article_title': 'Championship Victory in Overtime',
                    'source_category': 'sports'
                }
            ]
        else:
            # Generic test input generation based on input specifications
            for i in range(3):  # Generate 3 test cases
                test_input = {}
                for inp in workflow_inputs:
                    test_input[inp['name']] = self._generate_sample_value(inp)
                test_input_sets.append(test_input)
        
        return test_input_sets
    
    def _generate_sample_value(self, input_spec: Dict[str, Any]) -> Any:
        """
        Generate sample value based on input specification
        
        Args:
            input_spec: Input parameter specification
            
        Returns:
            Sample value for the input
        """
        input_type = input_spec.get('type', 'string')
        input_name = input_spec.get('name', 'input')
        
        if input_type == 'string':
            return f"sample_{input_name}_value"
        elif input_type == 'number':
            return 42
        elif input_type == 'boolean':
            return True
        elif input_type == 'array':
            return [f"item1_{input_name}", f"item2_{input_name}"]
        else:
            return f"sample_{input_name}"
    
    @retry_with_backoff(max_retries=2, delay=1.0)
    def _execute_single_workflow(self, workflow_json: Dict[str, Any], test_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single workflow with given input
        
        Args:
            workflow_json: Workflow definition
            test_input: Input data for execution
            
        Returns:
            Workflow execution result
        """
        # Import here to avoid circular dependencies
        from evoagentx.workflow.workflow_graph import WorkFlowGraph
        
        start_time = time.time()
        
        try:
            # Create workflow from JSON
            workflow = WorkFlowGraph.from_dict(workflow_json)
            
            # For now, simulate workflow execution since we need proper setup
            # In real implementation, you would call workflow.async_execute(test_input)
            execution_result = self._simulate_workflow_execution(workflow, test_input)
            
            execution_time = time.time() - start_time
            execution_result['execution_time'] = execution_time
            
            return execution_result
            
        except Exception as e:
            # Re-raise with additional context
            raise RuntimeError(f"Workflow execution failed: {str(e)}") from e
    
    def _simulate_workflow_execution(self, workflow, test_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate workflow execution for testing purposes
        TODO: Replace with actual workflow execution
        
        Args:
            workflow: Workflow object
            test_input: Input data
            
        Returns:
            Simulated execution result
        """
        # Simulate processing time
        time.sleep(0.1)
        
        # Generate mock output based on workflow outputs
        simulated_output = {}
        
        # For content analysis workflow, generate appropriate outputs
        if 'article_content' in test_input:
            simulated_output = {
                'categories': [{'name': 'technology', 'confidence': 0.85}],
                'key_topics': ['AI', 'trends', '2024'],
                'sentiment_score': 0.3,
                'summary': 'Article discusses technology trends and AI developments.'
            }
        else:
            # Generic output simulation
            simulated_output = {
                'result': 'processed_data',
                'status': 'completed'
            }
        
        return simulated_output
    
    def _analyze_errors(self, failed_executions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze error patterns in failed executions
        
        Args:
            failed_executions: List of failed execution results
            
        Returns:
            Error analysis summary
        """
        if not failed_executions:
            return {'total_errors': 0, 'error_types': {}}
        
        error_types = {}
        expected_errors = 0
        unexpected_errors = 0
        
        for execution in failed_executions:
            error_info = execution.get('error', {})
            error_type = error_info.get('type', 'Unknown')
            is_expected = error_info.get('is_expected', False)
            
            if error_type not in error_types:
                error_types[error_type] = {
                    'count': 0,
                    'expected_count': 0,
                    'unexpected_count': 0,
                    'examples': []
                }
            
            error_types[error_type]['count'] += 1
            if is_expected:
                error_types[error_type]['expected_count'] += 1
                expected_errors += 1
            else:
                error_types[error_type]['unexpected_count'] += 1
                unexpected_errors += 1
            
            # Store error message as example
            if len(error_types[error_type]['examples']) < 3:
                error_types[error_type]['examples'].append(error_info.get('message', ''))
        
        return {
            'total_errors': len(failed_executions),
            'expected_errors': expected_errors,
            'unexpected_errors': unexpected_errors,
            'error_types': error_types
        }


def execute_workflows_batch(
    test_batch: List[Dict[str, Any]], 
    config: EvaluationConfig
) -> Dict[str, Any]:
    """
    Execute a batch of workflows in a single process
    
    Args:
        test_batch: Batch of test data with workflow definitions
        config: Evaluation configuration
        
    Returns:
        Batch execution results
    """
    evaluator = ExecutionEvaluator(config)
    results = []
    errors = []
    
    start_time = time.time()
    
    for test_data in test_batch:
        try:
            result = evaluator.execute_workflow_with_test_inputs(test_data)
            results.append(result)
        except Exception as e:
            error_info = capture_exception_details(e)
            error_info['test_id'] = test_data.get('test_id', 'unknown')
            errors.append(error_info)
    
    execution_time = time.time() - start_time
    
    return {
        'test_results': results,
        'errors': errors,
        'summary': {
            'total_tests': len(test_batch),
            'successful': len(results),
            'failed': len(errors),
            'execution_time': execution_time
        }
    }


def run_layer_2_evaluation(
    layer_1_results: Dict[str, Any], 
    config: EvaluationConfig
) -> Dict[str, Any]:
    """
    Run Layer 2 execution evaluation with multiprocessing
    
    Args:
        layer_1_results: Results from Layer 1 evaluation
        config: Evaluation configuration
        
    Returns:
        Complete execution evaluation results
    """
    print(f"Starting Layer 2 evaluation...")
    
    # Extract successful workflows from Layer 1 results
    successful_workflows = [
        result for result in layer_1_results.get('test_results', [])
        if result.get('success', False) and 'workflow_json' in result
    ]
    
    if not successful_workflows:
        print("No successful workflows found from Layer 1 to execute")
        return {
            'layer': 2,
            'test_results': [],
            'errors': [],
            'summary': {'total_tests': 0, 'successful': 0, 'failed': 0},
            'message': 'No workflows available for execution testing'
        }
    
    print(f"Found {len(successful_workflows)} workflows to execute...")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(config.checkpoint_dir, 2)
    if checkpoint:
        print("Found existing checkpoint for Layer 2, resuming from last state...")
        completed_ids = {result['test_id'] for result in checkpoint.get('test_results', [])}
        successful_workflows = [
            workflow for workflow in successful_workflows 
            if workflow.get('test_id') not in completed_ids
        ]
        print(f"Resuming with {len(successful_workflows)} remaining workflows...")
    else:
        checkpoint = {'test_results': [], 'errors': [], 'summary': {}}
    
    if not successful_workflows:
        print("All workflows already executed for Layer 2")
        return checkpoint
    
    # Create batches for parallel processing
    batches = create_batch_iterator(successful_workflows, config.batch_size)
    all_results = []
    
    # Determine optimal number of processes
    num_processes = min(config.max_processes, len(batches))
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(execute_workflows_batch, batch, config): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_result = future.result()
                all_results.append(batch_result)
                print(f"Completed execution batch {batch_idx + 1}/{len(batches)}")
                
                # Save intermediate checkpoint
                if len(all_results) % 3 == 0:  # Save every 3 batches
                    intermediate_results = merge_results([checkpoint] + all_results)
                    save_checkpoint(config.checkpoint_dir, 2, intermediate_results)
                    
            except Exception as e:
                print(f"Execution batch {batch_idx + 1} failed: {e}")
                error_info = capture_exception_details(e)
                all_results.append({
                    'test_results': [],
                    'errors': [error_info],
                    'summary': {'total_tests': len(batches[batch_idx]), 'successful': 0, 'failed': 1}
                })
    
    # Merge all results
    final_results = merge_results([checkpoint] + all_results)
    final_results['layer'] = 2
    final_results['total_execution_time'] = time.time() - start_time
    final_results['timestamp'] = datetime.now().isoformat()
    
    # Save final checkpoint
    save_checkpoint(config.checkpoint_dir, 2, final_results)
    
    # Save final results
    results_file = os.path.join(config.results_dir, 'layer_2_execution_evaluation.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"Layer 2 evaluation completed in {final_results['total_execution_time']:.2f} seconds")
    print(f"Results saved to: {results_file}")
    
    return final_results
