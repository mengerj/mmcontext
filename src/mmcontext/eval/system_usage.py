import logging
import os
import platform
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import pynvml
import torch


class SystemMonitor:
    """A class for monitoring system resource usage over time.

    Parameters
    ----------
    interval : int, optional
        The interval in seconds between each monitoring update. Default is 1 second.
    gpu_idx : int, optional
        The index of the GPU to monitor. Default is 2. Because of IMBI GPU setup.
    """

    def __init__(self, interval=1, gpu_idx=None, logger=None):
        self.interval = interval
        self.gpu_indices = [gpu_idx] if isinstance(gpu_idx, int) else gpu_idx
        self.num_cpus = psutil.cpu_count(logical=True)
        self.cpu_usage = []
        self.cpu_per_core = []
        self.memory_usage = []
        self.disk_io = []
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.baseline_memory = psutil.virtual_memory().used / (1024**3)  # GB
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor)
        self.logger = logger or logging.getLogger(__name__)
        self.num_threads = []
        # self.cpu_affinity = []

        # GPU Monitoring Initialization
        self.gpu_available = False
        self.gpu_usage = []
        self.gpu_memory_usage = []
        self.gpu_name = None

        self._initialize_gpu_monitoring()

    def _initialize_gpu_monitoring(self):
        # Try NVIDIA GPU
        try:
            import pynvml

            pynvml.nvmlInit()
            self.gpu_handles = []
            self.gpu_names = []

            # First check CUDA_VISIBLE_DEVICES
            assigned_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if assigned_gpus:
                # Check if the entries are UUIDs or indices
                gpu_entries = assigned_gpus.split(",")
                for entry in gpu_entries:
                    entry = entry.strip()
                    try:
                        # Check if it's a numeric index
                        gpu_idx = int(entry)
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                        self.gpu_handles.append(handle)
                    except ValueError:
                        # Assume it's a UUID and fetch the handle
                        handle = pynvml.nvmlDeviceGetHandleByUUID(entry)
            else:
                # If no CUDA_VISIBLE_DEVICES, use gpu_indices or detect all available GPUs
                if self.gpu_indices is not None:
                    # Use specified GPU indices
                    for idx in self.gpu_indices:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        self.gpu_handles.append(handle)
                else:
                    # Detect all available GPUs
                    device_count = pynvml.nvmlDeviceGetCount()
                    for idx in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        self.gpu_handles.append(handle)

            # Get names and memory info for all detected GPUs
            for handle in self.gpu_handles:
                name = pynvml.nvmlDeviceGetName(handle)
                self.gpu_names.append(name.decode() if isinstance(name, bytes) else name)

            self.gpu_total_memory = [
                pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3) for handle in self.gpu_handles
            ]
            self.gpu_available = True
            self.gpu_type = "NVIDIA"
            self.logger.info(f"Detected NVIDIA GPUs: {self.gpu_names}")

        except Exception as e:
            self.logger.info(f"No NVIDIA GPU detected or pynvml not installed: {str(e)}")
            # Try detecting Apple GPU
            if platform.system() == "Darwin" and "macOS" in platform.platform():
                self.gpu_available = False
                self.gpu_type = "Apple"
                self.gpu_name = "Apple Integrated GPU"
                self.logger.info("Detected Apple GPU. But not supported for detailed monitoring.")
            else:
                self.logger.info("No supported GPU detected.")
                self.gpu_available = False

    def _monitor(self):
        process = psutil.Process()
        prev_disk_io_counters = None
        prev_time = time.time()
        while not self._stop_event.is_set():
            timestamp = time.time()
            interval_duration = timestamp - prev_time
            prev_time = timestamp
            # Measure CPU usage per core
            cpu_percents = psutil.cpu_percent(interval=self.interval, percpu=True)
            total_cpu_usage_percent = sum(cpu_percents)
            total_cpu_usage_cores = total_cpu_usage_percent / self.num_cpus
            self.cpu_usage.append((timestamp, total_cpu_usage_percent))
            self.cpu_per_core.append((timestamp, total_cpu_usage_cores))

            # Measure memory usage
            mem = psutil.virtual_memory()
            used_memory_gb = (mem.total - mem.available) / (1024**3)
            used_memory_gb -= self.baseline_memory
            self.memory_usage.append((timestamp, used_memory_gb))

            # Measure disk I/O
            disk_io_counters = psutil.disk_io_counters()
            if prev_disk_io_counters is not None:
                read_bytes = disk_io_counters.read_bytes - prev_disk_io_counters.read_bytes
                write_bytes = disk_io_counters.write_bytes - prev_disk_io_counters.write_bytes
                read_rate_mb_s = (read_bytes / (1024**2)) / interval_duration if interval_duration > 0 else 0
                write_rate_mb_s = (write_bytes / (1024**2)) / interval_duration if interval_duration > 0 else 0
            else:
                read_rate_mb_s = write_rate_mb_s = 0  # First iteration

            self.disk_io.append((timestamp, read_rate_mb_s, write_rate_mb_s))
            prev_disk_io_counters = disk_io_counters

            # Get number of threads and cpu core affinity
            self.num_threads.append((timestamp, process.num_threads()))
            # self.cpu_affinity.append((timestamp, process.cpu_affinity()))

            # GPU Monitoring
            if self.gpu_available:
                if self.gpu_type == "NVIDIA":
                    self._monitor_nvidia_gpu(timestamp)
                elif self.gpu_type == "Apple":
                    self._monitor_apple_gpu(timestamp)
                else:
                    pass  # Unsupported GPU type

    def _monitor_nvidia_gpu(self, timestamp):
        try:
            for idx, handle in enumerate(self.gpu_handles):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_usage_percent = util.gpu
                gpu_memory_used_gb = mem_info.used / (1024**3)
                if idx >= len(self.gpu_usage):
                    self.gpu_usage.append([])
                    self.gpu_memory_usage.append([])
                self.gpu_usage[idx].append((timestamp, gpu_usage_percent))
                self.gpu_memory_usage[idx].append((timestamp, gpu_memory_used_gb))
        except Exception as e:
            self.logger.error(f"Error monitoring NVIDIA GPU: {e}")

    def _monitor_apple_gpu(self, timestamp):
        # Limited monitoring available for Apple GPUs
        # Currently, we cannot get detailed GPU metrics on Apple Silicon using standard Python libraries
        # We'll log a placeholder or try using third-party tools if available
        self.gpu_usage.append((timestamp, None))
        self.gpu_memory_usage.append((timestamp, None))

    def start(self):
        """Start Monitoring System Resources."""
        self._thread.start()

    def stop(self):
        """Stop Monitoring System Resources."""
        self._stop_event.set()
        self._thread.join()
        # Shutdown GPU monitoring if necessary
        if self.gpu_available and self.gpu_type == "NVIDIA":
            import pynvml

            pynvml.nvmlShutdown()

    def summarize(self):
        """
        Summarizes the collected metrics

        Providing mean and maximum usage, core utilization efficiency, and disk I/O statistics.
        """
        summary = {}

        # CPU Usage
        total_cpu_usages = [usage for _, usage in self.cpu_usage]
        summary["cpu_usage_mean"] = sum(total_cpu_usages) / len(total_cpu_usages)
        summary["cpu_usage_max"] = max(total_cpu_usages)

        # Core Utilization
        core_utilizations = []
        for _, cores in self.cpu_per_core:
            core_utilizations.append(cores)
        summary["core_usage_mean"] = sum(core_utilizations) / len(core_utilizations)
        summary["core_usage_max"] = max(core_utilizations)

        # Memory Usage
        memory_usages = [usage for _, usage in self.memory_usage]
        summary["memory_usage_mean"] = sum(memory_usages) / len(memory_usages)
        summary["memory_usage_max"] = max(memory_usages)
        summary["total_memory"] = self.total_memory
        summary["baseline_memory"] = self.baseline_memory

        # Disk I/O
        read_rates = [read for _, read, _ in self.disk_io]
        write_rates = [write for _, _, write in self.disk_io]
        summary["disk_read_mb_s_mean"] = sum(read_rates) / len(read_rates)
        summary["disk_read_mb_s_max"] = max(read_rates)
        summary["disk_write_mb_s_mean"] = sum(write_rates) / len(write_rates)
        summary["disk_write_mb_s_max"] = max(write_rates)

        # GPU Usage
        if self.gpu_available:
            summary["total_gpu_memory"] = sum(self.gpu_total_memory)
            if self.gpu_type == "NVIDIA":
                summary["gpu_metrics"] = []
                for idx, (usage_data, memory_data, name, total_memory) in enumerate(
                    zip(self.gpu_usage, self.gpu_memory_usage, self.gpu_names, self.gpu_total_memory, strict=False)
                ):
                    gpu_summary = {}
                    gpu_usages = [usage for _, usage in usage_data if usage is not None]
                    gpu_memory_usages = [usage for _, usage in memory_data if usage is not None]

                    if gpu_usages:
                        gpu_summary["usage_mean"] = sum(gpu_usages) / len(gpu_usages)
                        gpu_summary["usage_max"] = max(gpu_usages)
                    if gpu_memory_usages:
                        gpu_summary["memory_usage_mean"] = sum(gpu_memory_usages) / len(gpu_memory_usages)
                        gpu_summary["memory_usage_max"] = max(gpu_memory_usages)

                    gpu_summary["name"] = name.decode() if isinstance(name, bytes) else name
                    gpu_summary["total_memory"] = total_memory
                    gpu_summary["gpu_id"] = idx

                    summary["gpu_metrics"].append(gpu_summary)
            else:
                # Handle Apple GPU or other types
                summary["gpu_metrics"] = [
                    {
                        "name": self.gpu_name,
                        "usage_mean": None,
                        "usage_max": None,
                        "memory_usage_mean": None,
                        "memory_usage_max": None,
                        "gpu_id": 0,
                    }
                ]
        else:
            summary["gpu_metrics"] = []

        return summary

    def print_summary(self):
        """Prints a formatted summary of the metrics."""
        summary = self.summarize()
        print("\nSystem Resource Usage Summary:")
        print(
            f"Core Utilization (mean/max % per core): {summary['core_usage_mean']:.2f}/{summary['core_usage_max']:.2f}% on {self.num_cpus} cores"
        )
        print(f"Memory Usage (mean/max GB): {summary['memory_usage_mean']:.2f}/{summary['memory_usage_max']:.2f} GB")
        print(f"Total System Memory: {summary['total_memory']:.2f} GB")
        print("Baseline Memory Usage: {:.2f} GB".format(summary["baseline_memory"]))
        print(
            f"Disk Read Rate (mean/max MB/s): {summary['disk_read_mb_s_mean']:.2f}/{summary['disk_read_mb_s_max']:.2f} MB/s"
        )
        print(
            f"Disk Write Rate (mean/max MB/s): {summary['disk_write_mb_s_mean']:.2f}/{summary['disk_write_mb_s_max']:.2f} MB/s"
        )
        if self.gpu_available:
            print("\nGPU Metrics:")
            print("Total GPU Memory: {:.2f} GB".format(summary["total_gpu_memory"]))
            for gpu in summary["gpu_metrics"]:
                print(f"\nGPU {gpu['gpu_id']}: {gpu['name']}")
                if gpu.get("usage_mean") is not None:
                    print(f"  Usage (mean/max %): {gpu['usage_mean']:.2f}/{gpu['usage_max']:.2f}%")
                if gpu.get("memory_usage_mean") is not None:
                    print(
                        f"  Memory Usage (mean/max GB): {gpu['memory_usage_mean']:.2f}/{gpu['memory_usage_max']:.2f} GB"
                    )
                    print(f"  Total Memory: {gpu['total_memory']:.2f} GB")
        else:
            print("\nNo supported GPU detected.")

    def save(self, save_dir):
        """Save the metrics as a csv file."""
        import pandas as pd

        name = "sys_metrics.csv"
        save_path = os.path.join(save_dir, name)
        summary = self.summarize()
        df = pd.DataFrame([summary])
        df.to_csv(save_path, index=False)
        self.logger.info(f"Metrics saved to {save_path}")

    def plot_metrics(self, save_dir=None):
        """
        Plots the collected metrics over time.

        If save_path is provided, saves the plots to the specified directory.
        """
        import os

        time_format = "%H:%M:%S"

        # Helper function to format x-axis ticks
        def format_time_ticks(timestamps):
            num_points = len(timestamps)
            max_labels = 10  # Maximum number of x-axis labels
            if num_points <= max_labels:
                # Use all timestamps as ticks
                tick_positions = range(num_points)
                tick_labels = [time.strftime(time_format, time.localtime(ts)) for ts in timestamps]
            else:
                # Select evenly spaced timestamps
                tick_positions = np.linspace(0, num_points - 1, max_labels, dtype=int)
                tick_labels = [time.strftime(time_format, time.localtime(timestamps[pos])) for pos in tick_positions]
            return tick_positions, tick_labels

        # CPU Usage Plot
        timestamps, cpu_usages = zip(*self.cpu_per_core, strict=False)
        tick_positions, tick_labels = format_time_ticks(timestamps)
        plt.figure()
        plt.plot(cpu_usages)
        plt.xlabel("Time")
        plt.ylabel("CPU Usage (avg % / core)")
        plt.title(f"Avg. CPU Usage Over Time from {self.num_cpus} Cores")
        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.ylim(0, 100)  # Set the y-axis limits from 0 to 100
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "cpu_usage.png"))
            plt.close()
        else:
            plt.show()

        # Memory Usage Plot
        timestamps, mem_usages = zip(*self.memory_usage, strict=False)
        tick_positions, tick_labels = format_time_ticks(timestamps)
        plt.figure()
        plt.plot(mem_usages)
        plt.xlabel("Time")
        plt.ylabel("Memory Usage (GB)")
        plt.title("Memory Usage Over Time")
        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "memory_usage.png"))
            plt.close()
        else:
            plt.show()

        # Disk I/O Plot
        timestamps, read_rates, write_rates = zip(*self.disk_io, strict=False)
        tick_positions, tick_labels = format_time_ticks(timestamps)
        plt.figure()
        plt.plot(read_rates, label="Read Rate (MB/s)")
        plt.plot(write_rates, label="Write Rate (MB/s)")
        plt.xlabel("Time")
        plt.ylabel("Disk I/O Rate (MB/s)")
        plt.title("Disk I/O Rates Over Time")
        plt.legend()
        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "disk_io.png"))
            plt.close()
        else:
            plt.show()

        # GPU Usage Plot - All GPUs on one plot
        if self.gpu_available and any(usage_data for usage_data in self.gpu_usage):
            plt.figure(figsize=(12, 6))
            max_rel_time = 0
            for idx, usage_data in enumerate(self.gpu_usage):
                if usage_data:
                    timestamps, gpu_usages = zip(*usage_data, strict=False)
                    rel_times = [t - timestamps[0] for t in timestamps]
                    max_rel_time = max(max_rel_time, rel_times[-1])
                    plt.plot(rel_times, gpu_usages, label=f"{self.gpu_names[idx]}")

            # Create evenly spaced tick marks (maximum 10)
            num_ticks = min(10, len(rel_times))
            tick_positions = np.linspace(0, max_rel_time, num_ticks)
            tick_labels = [f"{t:.0f}s" for t in tick_positions]

            plt.xlabel("Time (seconds)")
            plt.ylabel("GPU Usage (%)")
            plt.ylim(0, 101)
            plt.title("GPU Usage Over Time")
            plt.grid(True)
            plt.legend()
            plt.xticks(tick_positions, tick_labels)
            plt.tight_layout()

            if save_dir:
                plt.savefig(os.path.join(save_dir, "gpu_usage.png"))
                plt.close()
            else:
                plt.show()

        # GPU Memory Usage Plot - All GPUs on one plot
        if self.gpu_available and any(memory_data for memory_data in self.gpu_memory_usage):
            plt.figure(figsize=(12, 6))
            max_rel_time = 0
            for idx, memory_data in enumerate(self.gpu_memory_usage):
                if memory_data:
                    timestamps, gpu_mem_usages = zip(*memory_data, strict=False)
                    rel_times = [t - timestamps[0] for t in timestamps]
                    max_rel_time = max(max_rel_time, rel_times[-1])
                    plt.plot(rel_times, gpu_mem_usages, label=f"{self.gpu_names[idx]}")

            # Create evenly spaced tick marks (maximum 10)
            num_ticks = min(10, len(rel_times))
            tick_positions = np.linspace(0, max_rel_time, num_ticks)
            tick_labels = [f"{t:.0f}s" for t in tick_positions]

            plt.xlabel("Time (seconds)")
            plt.ylabel("GPU Memory Usage (GB)")
            plt.title("GPU Memory Usage Over Time")
            plt.grid(True)
            plt.legend()
            plt.xticks(tick_positions, tick_labels)
            plt.tight_layout()

            if save_dir:
                plt.savefig(os.path.join(save_dir, "gpu_memory_usage.png"))
                plt.close()
            else:
                plt.show()

        # Save or Show Plots
        if save_dir:
            self.logger.info(f"Plots saved to {save_dir}")
