# 如何在 Streamlit Cloud 中添加 Hugging Face Token

## 问题
调试信息显示 `hf_token` 没有在 Streamlit Secrets 中，只有 `gcp_service_account`。

## 解决步骤

### 1. 打开 Streamlit Cloud 应用设置
- 访问 https://share.streamlit.io/
- 登录你的账户
- 找到你的应用（Spotify-music-analysis）
- 点击应用名称进入应用页面

### 2. 进入 Settings（设置）
- 在应用页面，点击右上角的 **☰** (三个横线菜单)
- 选择 **"Settings"** 或 **"⚙️ Settings"**

### 3. 找到 Secrets 配置
- 在左侧菜单中找到 **"Secrets"** 选项
- 或者直接滚动到页面底部的 **"Secrets"** 部分

### 4. 编辑 Secrets
点击 **"Edit secrets"** 或 **"✏️ Edit"** 按钮

### 5. 添加 hf_token
在编辑器中，确保你的 secrets 格式如下：

```toml
[gcp_service_account]
type = "service_account"
project_id = "你的项目ID"
private_key_id = "你的private_key_id"
private_key = "-----BEGIN PRIVATE KEY-----\n你的私钥\n-----END PRIVATE KEY-----\n"
client_email = "你的client_email"
client_id = "你的client_id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "你的cert_url"

hf_token = "你的HuggingFace_token_在这里"
```

**重要提示：**
- `hf_token` 必须是**顶级键**（与 `[gcp_service_account]` 同级）
- 不要放在 `[gcp_service_account]` 里面
- 不要写成 `[hf_token]` 或 `[hf]`
- 格式：`hf_token = "你的token"`（注意等号两边有空格）

### 6. 保存
- 点击 **"Save"** 按钮
- 确认保存成功

### 7. 重启应用
- 回到应用页面
- 点击右上角 **☰** 菜单
- 选择 **"Reboot app"** 或 **"🔄 Reboot"**
- 等待应用重启（通常需要 10-30 秒）

### 8. 验证
- 刷新应用页面
- 进入 **"🤖 LLM Summary"** 标签页
- 应该看到 **"✅ Hugging Face Available"**
- 点击 **"🔍 Debug Info"** 展开，应该看到：
  - `Available secrets keys: ['gcp_service_account', 'hf_token']`
  - `✅ Token found via getattr: hf_你的token前10位...`
  - `✅ get_hf_token() returned: hf_你的token前10位...`

## 常见错误

### ❌ 错误格式 1：嵌套在 gcp_service_account 中
```toml
[gcp_service_account]
type = "..."
hf_token = "..."  # ❌ 错误！不要放在这里
```

### ❌ 错误格式 2：使用 section
```toml
[hf_token]  # ❌ 错误！不要用 section
token = "..."
```

### ✅ 正确格式：顶级键
```toml
[gcp_service_account]
type = "..."

hf_token = "..."  # ✅ 正确！顶级键
```

## 如果还是不行

1. **检查 token 是否有效**
   - 访问 https://huggingface.co/settings/tokens
   - 确认 token 存在且状态为 "Active"
   - 确认 token 有 "Read" 权限

2. **检查 Streamlit Cloud 日志**
   - 在应用设置中查看 "Logs"
   - 查找是否有 secrets 相关的错误

3. **尝试重新创建 token**
   - 在 Hugging Face 设置中删除旧 token
   - 创建新 token（确保有 "Read" 权限）
   - 在 Streamlit Cloud 中更新 secrets

4. **联系我**
   - 提供 Streamlit Cloud 的截图
   - 提供 Debug Info 的完整输出

