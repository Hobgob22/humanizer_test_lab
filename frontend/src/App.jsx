import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AuthGuard } from "./components/AuthGuard";
import { Login } from "./pages/Login";
import { JobStatus } from "./pages/JobStatus";
import { NewRun } from "./pages/NewRun";
import { BenchmarkAnalysis } from "./pages/BenchmarkAnalysis";
import { DocumentBrowser } from "./pages/DocumentBrowser";
import { PreviewResults } from "./pages/PreviewResults";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/*"
          element={
            <AuthGuard>
              <Layout>
                <Routes>
                  <Route path="/" element={<JobStatus />} />
                  <Route path="/new-run" element={<NewRun />} />
                  <Route path="/benchmark-analysis" element={<BenchmarkAnalysis />} />
                  <Route path="/document-browser" element={<DocumentBrowser />} />
                  <Route path="/preview-results" element={<PreviewResults />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Layout>
            </AuthGuard>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
