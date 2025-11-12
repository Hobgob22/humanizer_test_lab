import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

class APIClient {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        "Content-Type": "application/json",
      },
      timeout: 300000, // 5 minutes for large data loading
    });

    // Add auth token to requests if available
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem("auth_token");
      if (token) {
        config.headers["X-API-Key"] = token;
      }
      return config;
    });
  }

  // Auth
  async checkAuth(password) {
    const response = await this.client.post("/api/auth/check", { password });
    return response.data;
  }

  // Jobs
  async createJob(jobData) {
    const response = await this.client.post("/api/jobs/", jobData);
    return response.data;
  }

  async listJobs(limit = 50) {
    const response = await this.client.get(`/api/jobs/?limit=${limit}`);
    return response.data;
  }

  async getJob(jobId) {
    const response = await this.client.get(`/api/jobs/${jobId}`);
    return response.data;
  }

  async cancelJob(jobId) {
    const response = await this.client.post(`/api/jobs/${jobId}/cancel`);
    return response.data;
  }

  async getJobLogs(jobId) {
    const response = await this.client.get(`/api/jobs/${jobId}/logs`);
    return response.data;
  }

  // Runs
  async listRuns() {
    const response = await this.client.get("/api/runs/");
    return response.data;
  }

  async loadRun(runName) {
    const response = await this.client.get(`/api/runs/${encodeURIComponent(runName)}`);
    return response.data;
  }

  async deleteRun(runName) {
    const response = await this.client.delete(`/api/runs/${encodeURIComponent(runName)}`);
    return response.data;
  }

  // Documents
  async getDocuments(runName) {
    const response = await this.client.get(`/api/documents/${encodeURIComponent(runName)}`);
    return response.data;
  }

  async getDocument(runName, docName) {
    const response = await this.client.get(
      `/api/documents/${encodeURIComponent(runName)}/${encodeURIComponent(docName)}`
    );
    return response.data;
  }

  // Statistics
  async computeStatistics(runNames, merge = false) {
    const response = await this.client.post("/api/statistics/", {
      run_names: runNames,
      merge,
    });
    return response.data;
  }

  async getStatisticsStatus(taskId) {
    const response = await this.client.get(`/api/statistics/${taskId}`);
    return response.data;
  }

  // Health
  async healthCheck() {
    const response = await this.client.get("/api/health");
    return response.data;
  }

  // WebSocket
  createWebSocket() {
    const wsUrl = API_BASE_URL.replace("http", "ws");
    return new WebSocket(`${wsUrl}/api/ws`);
  }
}

export const api = new APIClient();
export default api;
