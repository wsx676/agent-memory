import { BrowserRouter, Route, Routes } from "react-router-dom";

import DashboardPage from "@/pages/DashboardPage";
import DeliveryPage from "@/pages/DeliveryPage";
import DocumentsPage from "@/pages/DocumentsPage";
import QualityPage from "@/pages/QualityPage";
import ResultsPage from "@/pages/ResultsPage";
import TasksPage from "@/pages/TasksPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/tasks" element={<TasksPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/quality" element={<QualityPage />} />
        <Route path="/delivery" element={<DeliveryPage />} />
      </Routes>
    </BrowserRouter>
  );
}
