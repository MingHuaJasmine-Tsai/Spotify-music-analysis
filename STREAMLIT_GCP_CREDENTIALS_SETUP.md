# Streamlit Cloud GCP 凭证配置指南

## 问题说明

Streamlit Cloud 不在 Google Cloud Platform 环境中运行，因此无法使用默认的 metadata service 来获取 GCP 凭证。需要在 Streamlit Secrets 中配置 GCP Service Account 凭证。

## 解决方案

代码已经更新，现在支持从 Streamlit Secrets 读取 GCP 凭证。

## 配置步骤

### 1. 创建 GCP Service Account Key

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 进入项目：`ba882-qstba-group7-fall2025`
3. 导航到 **IAM & Admin** → **Service Accounts**
4. 选择或创建一个 Service Account（需要有 Storage Object Viewer 权限）
5. 点击 Service Account → **Keys** → **Add Key** → **Create new key**
6. 选择 **JSON** 格式
7. 下载 JSON 文件

### 2. 在 Streamlit Cloud 中配置 Secrets

1. 访问 [Streamlit Cloud](https://share.streamlit.io/)
2. 进入你的应用
3. 点击 **⚙️ Settings** → **Secrets**
4. 在 Secrets 编辑器中添加以下内容：

```toml
[gcp_service_account]
type = "service_account"
project_id = "ba882-qstba-group7-fall2025"
private_key_id = "你的 private_key_id"
private_key = """-----BEGIN PRIVATE KEY-----
你的私钥内容
-----END PRIVATE KEY-----"""
client_email = "你的 service-account@project.iam.gserviceaccount.com"
client_id = "你的 client_id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "你的证书 URL"
```

**或者更简单的方法：**

直接将下载的 JSON 文件内容复制，然后转换为 TOML 格式：

```toml
[gcp_service_account]
type = "service_account"
project_id = "ba882-qstba-group7-fall2025"
private_key_id = "从 JSON 文件复制"
private_key = """从 JSON 文件复制（包括 BEGIN/END 标记）"""
client_email = "从 JSON 文件复制"
client_id = "从 JSON 文件复制"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "从 JSON 文件复制"
```

### 3. 确保 Service Account 有正确的权限

Service Account 需要以下权限：
- **Storage Object Viewer** - 读取 GCS bucket 中的文件
- 或者 **Storage Admin** - 完整访问权限（如果将来需要写入）

### 4. 重新部署应用

1. 保存 Secrets
2. 在 Streamlit Cloud 中点击 **☰** 菜单 → **Reboot app**
3. 等待应用重新启动

## 验证

应用重新启动后，应该能够：
- ✅ 成功连接到 GCS
- ✅ 加载 Silver Layer 数据
- ✅ 显示所有可视化图表

## 本地开发

本地开发时，代码会自动使用默认凭证（通过 `gcloud auth application-default login`），不需要配置 Secrets。

## 故障排除

如果仍然看到错误：

1. **检查 Service Account 权限**：
   - 确保有 Storage Object Viewer 权限
   - 确保可以访问 `gs://apidatabase/cleaned/` bucket

2. **检查 Secrets 格式**：
   - 确保 `private_key` 包含完整的 BEGIN/END 标记
   - 确保所有字段都正确填写

3. **检查应用日志**：
   - 在 Streamlit Cloud 中查看 Logs
   - 查找具体的错误信息

## 安全提示

⚠️ **重要**：Service Account Key 包含敏感信息，请：
- 不要将 JSON 文件提交到 Git
- 不要分享 Service Account Key
- 定期轮换密钥
- 使用最小权限原则

