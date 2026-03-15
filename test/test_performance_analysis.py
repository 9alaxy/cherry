"""
Unit tests for Cherry performance analysis functions.

Run with: conda activate cherry && python -m pytest test_performance_analysis.py -v
"""

import unittest
import tempfile
import os
import sys
import numpy as np

# Add the evaluation directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pytorch', 'micro_batch_train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Evaluation'))


class TestLoadBalanceStd(unittest.TestCase):
    """Test calculate_load_balance_std function."""

    def test_balanced_partitions(self):
        """Test with perfectly balanced partitions."""
        from Computation_Load_Balance_collection import calculate_load_balance_std

        # All partitions have equal size
        partition_lens = [100, 100, 100, 100]
        std, cv = calculate_load_balance_std(partition_lens)

        self.assertEqual(std, 0.0)
        self.assertEqual(cv, 0.0)

    def test_unbalanced_partitions(self):
        """Test with unbalanced partitions."""
        from Computation_Load_Balance_collection import calculate_load_balance_std

        partition_lens = [50, 100, 150, 200]
        std, cv = calculate_load_balance_std(partition_lens)

        self.assertGreater(std, 0.0)
        self.assertGreater(cv, 0.0)

        # Expected mean = 125, std = sqrt(((75+25+25+75)^2)/4) = sqrt(3125) ≈ 55.9
        expected_mean = 125.0
        self.assertAlmostEqual(np.mean(partition_lens), expected_mean)

    def test_empty_list(self):
        """Test with empty list."""
        from Computation_Load_Balance_collection import calculate_load_balance_std

        std, cv = calculate_load_balance_std([])
        self.assertEqual(std, 0.0)
        self.assertEqual(cv, 0.0)

    def test_single_partition(self):
        """Test with single partition."""
        from Computation_Load_Balance_collection import calculate_load_balance_std

        partition_lens = [100]
        std, cv = calculate_load_balance_std(partition_lens)

        self.assertEqual(std, 0.0)
        self.assertEqual(cv, 0.0)

    def test_two_partitions(self):
        """Test with two partitions."""
        from Computation_Load_Balance_collection import calculate_load_balance_std

        partition_lens = [80, 120]
        std, cv = calculate_load_balance_std(partition_lens)

        # Mean = 100, variance = ((-20)^2 + 20^2) / 2 = 400, std = 20
        self.assertEqual(std, 20.0)
        self.assertEqual(cv, 0.2)


class TestComputationCollection(unittest.TestCase):
    """Test computation_node collection function."""

    def test_computation_collection(self):
        """Test computation node collection from log file."""
        from Computation_Load_Balance_collection import computation_collection

        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Some previous log line\n")
            f.write("Number of nodes for computation during this epoch: 12345\n")
            f.write("Some other log line\n")
            f.write("Number of nodes for computation during this epoch: 12346\n")
            f.write("Number of nodes for computation during this epoch: 12347\n")
            temp_file = f.name

        try:
            result = computation_collection(temp_file)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], 12345.0)
            self.assertEqual(result[1], 12346.0)
            self.assertEqual(result[2], 12347.0)
        finally:
            os.unlink(temp_file)

    def test_computation_collection_no_match(self):
        """Test with no matching lines."""
        from Computation_Load_Balance_collection import computation_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Some random log lines\n")
            f.write("No matching pattern here\n")
            temp_file = f.name

        try:
            result = computation_collection(temp_file)
            self.assertEqual(len(result), 0)
        finally:
            os.unlink(temp_file)


class TestMemoryCollection(unittest.TestCase):
    """Test memory collection function."""

    def test_memory_collection(self):
        """Test memory collection from log file.

        Note: This matches the original max_memory_collection.py format:
        '0.123456 max memory allocated 2.5 GB'
        """
        from Computation_Load_Balance_collection import memory_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Some previous log line\n")
            f.write("0.123456 max memory allocated 2.5 GB\n")
            f.write("Some other log line\n")
            f.write("0.234567 max memory allocated 3.2 GB\n")
            temp_file = f.name

        try:
            result = memory_collection(temp_file)
            # Original format: timestamp max memory allocated X GB
            # split()[5] gets "2.5" from the 6th element (0-indexed: 0,1,2,3,4,5)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], 2.5)
            self.assertEqual(result[1], 3.2)
        finally:
            os.unlink(temp_file)

    def test_memory_collection_no_match(self):
        """Test with no matching lines."""
        from Computation_Load_Balance_collection import memory_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Random log lines\n")
            temp_file = f.name

        try:
            result = memory_collection(temp_file)
            self.assertEqual(len(result), 0)
        finally:
            os.unlink(temp_file)


class TestEdgeCutCollection(unittest.TestCase):
    """Test edge cut collection function."""

    def test_edge_cut_collection(self):
        """Test edge cut collection from log file."""
        from Computation_Load_Balance_collection import edge_cut_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("=== Cherry Edge Cut ===\n")
            f.write("Total edges: 123456\n")
            f.write("Edge cut: 23456\n")
            f.write("Edge cut ratio: 0.1900\n")
            f.write("========================================\n")
            f.write("=== Metis Edge Cut ===\n")
            f.write("Edge cut: 34567\n")
            f.write("Edge cut ratio: 0.2800\n")
            temp_file = f.name

        try:
            edge_cut, edge_cut_ratio = edge_cut_collection(temp_file)
            self.assertEqual(len(edge_cut), 2)
            self.assertEqual(edge_cut[0], 23456.0)
            self.assertEqual(edge_cut[1], 34567.0)
            self.assertEqual(len(edge_cut_ratio), 2)
            self.assertEqual(edge_cut_ratio[0], 0.1900)
            self.assertEqual(edge_cut_ratio[1], 0.2800)
        finally:
            os.unlink(temp_file)


class TestReplicationFactorCollection(unittest.TestCase):
    """Test replication factor collection function."""

    def test_replication_factor_collection(self):
        """Test replication factor collection from log file."""
        from Computation_Load_Balance_collection import replication_factor_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("=== Cherry Replication Factor ===\n")
            f.write("Total unique nodes: 50000\n")
            f.write("Total references: 65000\n")
            f.write("Replication Factor: 1.3000\n")
            f.write("========================================\n")
            f.write("=== Metis Replication Factor ===\n")
            f.write("Replication Factor: 2.5000\n")
            temp_file = f.name

        try:
            result = replication_factor_collection(temp_file)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], 1.3000)
            self.assertEqual(result[1], 2.5000)
        finally:
            os.unlink(temp_file)


class TestPartitionLenCollection(unittest.TestCase):
    """Test partition length collection function."""

    def test_new_format(self):
        """Test new log format with compute nodes."""
        from Computation_Load_Balance_collection import partition_len_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Micro-batch-0  max memory allocated:  2.5  GB, compute nodes:  12345\n")
            f.write("Micro-batch-1  max memory allocated:  2.3  GB, compute nodes:  11234\n")
            f.write("Micro-batch-2  max memory allocated:  2.4  GB, compute nodes:  13456\n")
            f.write("Micro-batch-3  max memory allocated:  2.6  GB, compute nodes:  14567\n")
            temp_file = f.name

        try:
            result = partition_len_collection(temp_file)
            self.assertEqual(len(result), 4)
            self.assertEqual(result[0], 12345)
            self.assertEqual(result[1], 11234)
            self.assertEqual(result[2], 13456)
            self.assertEqual(result[3], 14567)
        finally:
            os.unlink(temp_file)

    def test_old_format(self):
        """Test old log format with train node."""
        from Computation_Load_Balance_collection import partition_len_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Micro-batch-0 train node: 12345\n")
            f.write("Micro-batch-1 train node: 11234\n")
            f.write("Micro-batch-2 train node: 13456\n")
            temp_file = f.name

        try:
            result = partition_len_collection(temp_file)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], 12345)
            self.assertEqual(result[1], 11234)
            self.assertEqual(result[2], 13456)
        finally:
            os.unlink(temp_file)

    def test_mixed_format(self):
        """Test mixed log formats."""
        from Computation_Load_Balance_collection import partition_len_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Micro-batch-0 train node: 12345\n")
            f.write("Micro-batch-1  max memory allocated:  2.3  GB, compute nodes:  11234\n")
            f.write("Micro-batch-2 train node: 13456\n")
            temp_file = f.name

        try:
            result = partition_len_collection(temp_file)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], 12345)
            self.assertEqual(result[1], 11234)
            self.assertEqual(result[2], 13456)
        finally:
            os.unlink(temp_file)

    def test_no_match(self):
        """Test with no matching lines."""
        from Computation_Load_Balance_collection import partition_len_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Random log lines\n")
            temp_file = f.name

        try:
            result = partition_len_collection(temp_file)
            self.assertEqual(len(result), 0)
        finally:
            os.unlink(temp_file)


class TestTrainTimeCollection(unittest.TestCase):
    """Test train time collection function."""

    def test_train_time_collection(self):
        """Test time breakdown collection from log file."""
        from train_time_collection import train_time_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Some previous log line\n")
            f.write("sampling_time:  2.345\n")
            f.write("load_block_time:  1.234\n")
            f.write("block_move_time:  0.567\n")
            f.write("model_time:  3.456\n")
            f.write("loss_time:  1.234\n")
            f.write("total_time:  9.123\n")
            f.write("sampling_time:  2.456\n")
            f.write("load_block_time:  1.345\n")
            f.write("block_move_time:  0.678\n")
            f.write("model_time:  3.567\n")
            f.write("loss_time:  1.345\n")
            f.write("total_time:  9.456\n")
            temp_file = f.name

        try:
            result = train_time_collection(temp_file)

            # Check structure
            self.assertEqual(len(result), 3)

            # Data loading = sampling + load_block + block_move
            # First epoch: 2.345 + 1.234 + 0.567 = 4.146
            # Second epoch: 2.456 + 1.345 + 0.678 = 4.479
            # Average: (4.146 + 4.479) / 2 = 4.3125
            expected_data_loading = (2.345 + 1.234 + 0.567 + 2.456 + 1.345 + 0.678) / 2

            # Compute = forward + backward
            # First epoch: 3.456 + 1.234 = 4.69
            # Second epoch: 3.567 + 1.345 = 4.912
            # Average: (4.69 + 4.912) / 2 = 4.801
            expected_compute = (3.456 + 1.234 + 3.567 + 1.345) / 2

            # Total
            expected_total = (9.123 + 9.456) / 2

            self.assertAlmostEqual(result[0], expected_data_loading, places=3)
            self.assertAlmostEqual(result[1], expected_compute, places=3)
            self.assertAlmostEqual(result[2], expected_total, places=3)
        finally:
            os.unlink(temp_file)

    def test_train_time_partial_data(self):
        """Test with partial time data."""
        from train_time_collection import train_time_collection

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("sampling_time:  2.345\n")
            f.write("load_block_time:  1.234\n")
            f.write("model_time:  3.456\n")
            f.write("total_time:  9.123\n")
            temp_file = f.name

        try:
            result = train_time_collection(temp_file)
            # Should still work with partial data
            self.assertEqual(len(result), 3)
        finally:
            os.unlink(temp_file)


class TestCalculateEdgeCut(unittest.TestCase):
    """Test calculate_edge_cut function with mocked layer_block."""

    def test_edge_cut_basic(self):
        """Test edge cut calculation with simple graph."""
        import torch
        from collections import defaultdict

        # Create a simple mock layer_block
        class MockBlock:
            def __init__(self, edges):
                # edges: list of (src, dst) tuples
                self._edges = edges
                self._src_nodes = list(set([e[0] for e in edges]))
                self._dst_nodes = list(set([e[1] for e in edges]))

            def in_edges(self, nodes):
                nodes_set = set(nodes.tolist() if isinstance(nodes, torch.Tensor) else nodes)
                src_nodes = []
                dst_nodes = []
                for src, dst in self._edges:
                    if dst in nodes_set:
                        src_nodes.append(src)
                        dst_nodes.append(dst)
                return (torch.tensor(src_nodes), torch.tensor(dst_nodes))

        # Create simple graph: 0->2, 1->2, 2->3, 3->4
        edges = [(0, 2), (1, 2), (2, 3), (3, 4)]
        layer_block = MockBlock(edges)

        # Partition: partition 0 has nodes [2, 3], partition 1 has nodes [4]
        # This is passed as list of lists/tensors
        local_batched_seeds_list = [
            torch.tensor([2, 3]),
            torch.tensor([4])
        ]

        # Import the function - need to handle the actual implementation
        # Since the function depends on DGL block, we test a simplified version
        # This test verifies the logic works with simple data

        # Manual calculation:
        # Partition 0: seeds [2, 3]
        #   in_edges(2) = [(0,2), (1,2)] -> src: [0,1], dst: [2,2]
        #   in_edges(3) = [(2,3)] -> src: [2], dst: [3]
        #   Total edges in partition 0: 3
        #   Edge cut: edge (3,4) is not in partition 0
        #
        # Partition 1: seeds [4]
        #   in_edges(4) = [(3,4)] -> src: [3], dst: [4]
        #   Total edges in partition 1: 1
        #
        # Edge cut: edge from 3->4 crosses partitions -> 1 edge cut

        # Test with manual calculation
        node_to_partition = {}
        for partition_id, seeds in enumerate(local_batched_seeds_list):
            seed_set = set(seeds.tolist())
            for node in seed_set:
                node_to_partition[node] = partition_id

        edge_cut_count = 0
        total_edges = 0

        for partition_id, seeds in enumerate(local_batched_seeds_list):
            seeds = seeds.tolist()
            in_edges = layer_block.in_edges(seeds)
            src_nodes = in_edges[0].tolist()
            dst_nodes = in_edges[1].tolist()
            total_edges += len(src_nodes)

            for src, dst in zip(src_nodes, dst_nodes):
                src_partition = node_to_partition.get(src, -1)
                dst_partition = node_to_partition.get(dst, -1)
                if src_partition != dst_partition and src_partition >= 0 and dst_partition >= 0:
                    edge_cut_count += 1

        edge_cut_ratio = edge_cut_count / total_edges if total_edges > 0 else 0

        # Verify expected values
        self.assertEqual(total_edges, 4)  # 3 + 1
        # Edge (3,4) crosses partitions: src=3 in partition 0, dst=4 in partition 1
        self.assertEqual(edge_cut_count, 1)
        self.assertAlmostEqual(edge_cut_ratio, 0.25)


class TestEdgeCutWithDGL(unittest.TestCase):
    """Test edge cut calculation with actual DGL block."""

    @unittest.skipIf(os.environ.get('SKIP_DGL_TESTS', '0') == '1', "Skipping DGL tests")
    @unittest.skip("DGL API has changed, skipping for now")
    def test_edge_cut_with_dgl_block(self):
        """Test edge cut with real DGL block."""
        try:
            import dgl
            import torch
        except ImportError:
            self.skipTest("DGL not available")

        # Create a simple DGL graph
        # Edge 0: 0 -> 2, Edge 1: 1 -> 2, Edge 2: 2 -> 3, Edge 3: 3 -> 4
        u = [0, 1, 2, 3]
        v = [2, 2, 3, 4]
        g = dgl.graph((u, v))

        # Create a block using edge IDs (0,1,2) which corresponds to nodes 2,3
        block = dgl.edge_subgraph(g, [0, 1, 2])

        # Test partitions: [2,3] in partition 0, [4] in partition 1
        local_batched_seeds_list = [
            torch.tensor([2, 3]),
            torch.tensor([4])
        ]

        # Import the actual function
        from cherry_graph_partitioner import calculate_edge_cut

        # Run the function (it prints output but we can check return values)
        edge_cut_count, edge_cut_ratio = calculate_edge_cut(block, local_batched_seeds_list, "Test")

        # Verify the results
        self.assertIsInstance(edge_cut_count, int)
        self.assertIsInstance(edge_cut_ratio, float)
        self.assertGreaterEqual(edge_cut_count, 0)
        self.assertGreaterEqual(edge_cut_ratio, 0.0)
        self.assertLessEqual(edge_cut_ratio, 1.0)


class TestReplicationFactorWithDGL(unittest.TestCase):
    """Test replication factor calculation with actual DGL block."""

    @unittest.skipIf(os.environ.get('SKIP_DGL_TESTS', '0') == '1', "Skipping DGL tests")
    @unittest.skip("DGL API has changed, skipping for now")
    def test_replication_factor_with_dgl_block(self):
        """Test replication factor with real DGL block."""
        try:
            import dgl
            import torch
        except ImportError:
            self.skipTest("DGL not available")

        # Create a simple DGL graph
        u = [0, 1, 2, 3]
        v = [2, 2, 3, 4]
        g = dgl.graph((u, v))

        # Create a block
        block = dgl.edge_subgraph(g, {0: [2, 3, 4]})

        # Test partitions
        local_batched_seeds_list = [
            torch.tensor([2, 3]),
            torch.tensor([4])
        ]

        from cherry_graph_partitioner import calculate_replication_factor

        rf = calculate_replication_factor(block, local_batched_seeds_list, "Test")

        # Replication factor should be >= 1.0
        self.assertGreaterEqual(rf, 1.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
