import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function formatTimestamp(timestamp) {
  if (!timestamp) return "N/A";
  const date = new Date(timestamp * 1000);
  return date.toLocaleString();
}

export function formatDuration(seconds) {
  if (!seconds) return "0s";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

export function formatNumber(num, decimals = 2) {
  if (num === null || num === undefined) return "N/A";
  return Number(num).toFixed(decimals);
}

export function formatPercent(num, decimals = 1) {
  if (num === null || num === undefined) return "N/A";
  return `${(num * 100).toFixed(decimals)}%`;
}

export function getStatusColor(status) {
  const colors = {
    pending: "bg-yellow-500/10 text-yellow-700 border-yellow-500/20",
    running: "bg-blue-500/10 text-blue-700 border-blue-500/20",
    completed: "bg-green-500/10 text-green-700 border-green-500/20",
    failed: "bg-red-500/10 text-red-700 border-red-500/20",
    cancelled: "bg-gray-500/10 text-gray-700 border-gray-500/20",
  };
  return colors[status] || colors.pending;
}

export function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
