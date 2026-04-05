import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";

const app = express();
const port = 8080;
const target = process.env.DASHBOARD_API || "http://localhost:5000";

app.use("/api", createProxyMiddleware({ target, changeOrigin: true }));
app.use(express.static("public"));

app.listen(port, () => {
  console.log(`api-gateway listening on ${port}`);
});
