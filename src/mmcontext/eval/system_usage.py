import logging
import platform
import threading
import time

import numpy as np
import psutil


class SystemMonitor:
    """A class for monitoring system resource usage over time."""

    def __init__(self, interval=1):
        self.interval = interval
        self.num_cpus = psutil.cpu_count(logical=True)
        self.cpu_usage = []
        self.cpu_per_core = []
        self.memory_usage = []
        self.disk_io = []
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor)
        self.logger = logging.getLogger(__name__)
        self.num_threads = []
        self.cpu_affinity = []

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
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_total_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).total / (1024**3)  # GB
            self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode()
            self.gpu_available = True
            self.gpu_type = "NVIDIA"
            self.logger.info(f"Detected NVIDIA GPU: {self.gpu_name}")
        except Exception:
            self.logger.info("No NVIDIA GPU detected or pynvml not installed.")
            # Try detecting Apple GPU
            if platform.system() == "Darwin" and "Apple" in platform.platform():
                self.gpu_available = True
                self.gpu_type = "Apple"
                self.gpu_name = "Apple Integrated GPU"
                self.logger.info("Detected Apple GPU.")
            else:
                self.logger.info("No supported GPU detected.")
                self.gpu_available = False

    def _monitor(self):
        process = psutil.Process()
        prev_disk_io_counters = psutil.disk_io_counters()
        prev_time = time.time()
        while not self._stop_event.is_set():
            timestamp = time.time()
            interval_duration = timestamp - prev_time
            prev_time = timestamp
            # Measure CPU usage per core
            cpu_percents = psutil.cpu_percent(interval=self.interval, percpu=True)
            total_cpu_usage_percent = sum(cpu_percents)
            total_cpu_usage_cores = total_cpu_usage_percent / self.num_cpus
            self.cpu_usage.append((timestamp, total_cpu_usage_cores))
            self.cpu_per_core.append((timestamp, cpu_percents))

            # Measure memory usage
            mem = psutil.virtual_memory()
            used_memory_gb = (mem.total - mem.available) / (1024**3)
            self.memory_usage.append((timestamp, used_memory_gb))

            # Measure disk I/O
            disk_io_counters = psutil.disk_io_counters()
            read_bytes = disk_io_counters.read_bytes - prev_disk_io_counters.read_bytes
            write_bytes = disk_io_counters.write_bytes - prev_disk_io_counters.write_bytes
            if interval_duration > 0:
                read_rate_mb_s = (read_bytes / (1024**2)) / interval_duration
                write_rate_mb_s = (write_bytes / (1024**2)) / interval_duration
            else:
                read_rate_mb_s = 0
                write_rate_mb_s = 0
            self.disk_io.append((timestamp, read_rate_mb_s, write_rate_mb_s))
            prev_disk_io_counters = disk_io_counters

            # Get number of threads and cpu core affinity
            self.num_threads.append((timestamp, process.num_threads()))
            self.cpu_affinity.append((timestamp, process.cpu_affinity()))

            # GPU Monitoring
            if self.gpu_available:
                if self.gpu_type == "NVIDIA":
                    self._monitor_nvidia_gpu(timestamp)
                elif self.gpu_type == "Apple":
                    self._monitor_apple_gpu(timestamp)
                else:
                    pass  # Unsupported GPU type

    def _monitor_nvidia_gpu(self, timestamp):
        import pynvml

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_usage_percent = util.gpu
            gpu_memory_used_gb = mem_info.used / (1024**3)
            self.gpu_usage.append((timestamp, gpu_usage_percent))
            self.gpu_memory_usage.append((timestamp, gpu_memory_used_gb))
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
            core_utilizations.extend(cores)
        summary["core_usage_mean"] = sum(core_utilizations) / len(core_utilizations)
        summary["core_usage_max"] = max(core_utilizations)

        # Memory Usage
        memory_usages = [usage for _, usage in self.memory_usage]
        summary["memory_usage_mean"] = sum(memory_usages) / len(memory_usages)
        summary["memory_usage_max"] = max(memory_usages)
        summary["total_memory"] = self.total_memory

        # Disk I/O
        read_rates = [read for _, read, _ in self.disk_io]
        write_rates = [write for _, _, write in self.disk_io]
        summary["disk_read_mb_s_mean"] = sum(read_rates) / len(read_rates)
        summary["disk_read_mb_s_max"] = max(read_rates)
        summary["disk_write_mb_s_mean"] = sum(write_rates) / len(write_rates)
        summary["disk_write_mb_s_max"] = max(write_rates)

        # GPU Usage
        if self.gpu_available:
            gpu_usages = [usage for _, usage in self.gpu_usage if usage is not None]
            gpu_memory_usages = [usage for _, usage in self.gpu_memory_usage if usage is not None]
            if gpu_usages:
                summary["gpu_usage_mean"] = sum(gpu_usages) / len(gpu_usages)
                summary["gpu_usage_max"] = max(gpu_usages)
            if gpu_memory_usages:
                summary["gpu_memory_usage_mean"] = sum(gpu_memory_usages) / len(gpu_memory_usages)
                summary["gpu_memory_usage_max"] = max(gpu_memory_usages)
                summary["gpu_total_memory"] = self.gpu_total_memory
            summary["gpu_name"] = self.gpu_name
        else:
            summary["gpu_usage_mean"] = None
            summary["gpu_usage_max"] = None

        # Number of Threads
        num_threads = [threads for _, threads in self.num_threads]
        summary["num_threads_mean"] = sum(num_threads) / len(num_threads)
        summary["num_threads_max"] = max(num_threads)

        return summary

    def print_summary(self):
        """Prints a formatted summary of the metrics."""
        summary = self.summarize()
        print("\nSystem Resource Usage Summary:")
        print(f"CPU Usage (mean/max cores): {summary['cpu_usage_mean']:.2f}/{summary['cpu_usage_max']:.2f}")
        print(
            f"Core Utilization (mean/max % per core): {summary['core_usage_mean']:.2f}/{summary['core_usage_max']:.2f}%"
        )
        print(f"Memory Usage (mean/max GB): {summary['memory_usage_mean']:.2f}/{summary['memory_usage_max']:.2f} GB")
        print(f"Total System Memory: {summary['total_memory']:.2f} GB")
        print(
            f"Disk Read Rate (mean/max MB/s): {summary['disk_read_mb_s_mean']:.2f}/{summary['disk_read_mb_s_max']:.2f} MB/s"
        )
        print(
            f"Disk Write Rate (mean/max MB/s): {summary['disk_write_mb_s_mean']:.2f}/{summary['disk_write_mb_s_max']:.2f} MB/s"
        )
        if self.gpu_available:
            print(f"GPU Name: {summary['gpu_name']}")
            if summary["gpu_usage_mean"] is not None:
                print(f"GPU Usage (mean/max %): {summary['gpu_usage_mean']:.2f}/{summary['gpu_usage_max']:.2f}%")
            if summary["gpu_memory_usage_mean"] is not None:
                print(
                    f"GPU Memory Usage (mean/max GB): {summary['gpu_memory_usage_mean']:.2f}/{summary['gpu_memory_usage_max']:.2f} GB"
                )
                print(f"GPU Total Memory: {summary['gpu_total_memory']:.2f} GB")
        else:
            print("No supported GPU detected.")
        print(f"Number of Threads (mean/max): {summary['num_threads_mean']:.2f}/{summary['num_threads_max']}")

    def plot_metrics(self, save_path=None):
        """
        Plots the collected metrics over time.

        If save_path is provided, saves the plots to the specified directory.
        """
        import os

        import matplotlib.pyplot as plt

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
        timestamps, cpu_usages = zip(*self.cpu_usage, strict=False)
        tick_positions, tick_labels = format_time_ticks(timestamps)
        plt.figure()
        plt.plot(cpu_usages)
        plt.xlabel("Time")
        plt.ylabel("CPU Usage (avg % / core)")
        plt.title("Total CPU Usage Over Time")
        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "cpu_usage.png"))
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
        if save_path:
            plt.savefig(os.path.join(save_path, "memory_usage.png"))
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
        if save_path:
            plt.savefig(os.path.join(save_path, "disk_io.png"))
            plt.close()
        else:
            plt.show()

        # GPU Usage Plot
        if self.gpu_available and any(usage is not None for _, usage in self.gpu_usage):
            timestamps, gpu_usages = zip(*self.gpu_usage, strict=False)
            gpu_usages = [usage if usage is not None else 0 for usage in gpu_usages]
            tick_positions, tick_labels = format_time_ticks(timestamps)
            plt.figure()
            plt.plot(gpu_usages)
            plt.xlabel("Time")
            plt.ylabel("GPU Usage (%)")
            plt.title("GPU Usage Over Time")
            plt.xticks(tick_positions, tick_labels, rotation=45)
            plt.tight_layout()
            if save_path:
                plt.savefig(os.path.join(save_path, "gpu_usage.png"))
                plt.close()
            else:
                plt.show()

        # Save or Show Plots
        if save_path:
            self.logger.info(f"Plots saved to {save_path}")
