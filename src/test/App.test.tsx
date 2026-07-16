import { render, screen } from "@testing-library/react";

import App from "@/App";

test("renders dashboard shell title", () => {
  window.history.pushState({}, "控制台", "/");
  render(<App />);
  expect(screen.getByText("金融长文档智能阅读理解系统")).toBeInTheDocument();
  expect(screen.getByText("项目控制台")).toBeInTheDocument();
});
