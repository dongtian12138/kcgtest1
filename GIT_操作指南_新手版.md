# Git 新手操作指南（kcgtest1 专用）

> 你不用学完整的 git。这份指南只写你日常会用到的 6 条命令，
> 按"理解 → 日常 → 换机 → 同步 → 报错"的顺序讲，5 分钟看完。

---

## 一、一句话理解 git

- **仓库**：就是你的工程文件夹，git 会给它加一本"自动存档本"（藏在 .git 目录里）。
- **提交（commit）**：每改完一轮，拍一张快照存进存档本。以后随时能翻回任何一张。
- **标签（tag）**：给特别重要的存档起个名字，比如 `v20260814-backup`（已建好）。
- **远程（remote）**：把存档本放到 GitHub 上，另一台电脑就能拉过去。

**你的仓库现状**：已经建好，历史里有 2 个提交 + 1 个标签。
远程已指向 `https://github.com/dongtian12138/kcgtest1.git`（私有仓库）。
`artifacts/` 等大文件按 .gitignore 排除，**不会**进 git，仍走 U 盘。

---

## 二、日常六条命令（这台电脑上改完代码后）

```bash
cd ~/WorkPlace/kcgtest1

git status                                  # 1) 看看改了哪些文件（可随时跑，无害）
git add .                                   # 2) 把改动装进"待存档"筐
git commit -m "一句话：改了什么"             # 3) 存档
git log --oneline                           # 4) 看存档历史（按 q 退出）
git tag -a v20260820-milestone -m "说明"    # 5) 里程碑打标签（每通过一个重要验收就做一次）
git push                                    # 6) 推到 GitHub
```

**节奏建议**：每改完一个能跑通的小改动就 commit 一次（1、2、3 连做）；
不要攒十天半个月再存。commit 写清楚"改了什么、为什么"。

---

## 三、换新电脑怎么拿代码（两种方式）

### 方式 A：从 U 盘离线包拿（不需要网络）

```bash
# bundle 文件在便携目录里：kcgtest1_src_20260814.bundle（约 29M）
git clone ~/kcgtest1_src_20260814.bundle ~/WorkPlace/kcgtest1
cd ~/WorkPlace/kcgtest1
bash scripts/bootstrap.sh        # 装环境 + 构建
bash scripts/verify_all.sh       # 验收（详见 AGENT_README.md）
```

注意：bundle 里只有源码/文档/脚本；`artifacts/` 实验产物仍需从 U 盘的
核心包/完整包解压后放回 `~/WorkPlace/kcgtest1/artifacts`。

### 方式 B：从 GitHub 远程拉（需要网络）

```bash
git clone https://github.com/dongtian12138/kcgtest1.git ~/WorkPlace/kcgtest1
```

---

## 四、两台电脑长期同步

### 第一步：在 GitHub 建私有仓库（只做一次）

1. 登录 github.com → 右上角 **+** → **New repository**；
2. Repository name 填 `kcgtest1`；
3. 选 **Private**（私有），**不要**勾 "Add a README file"；
4. 点 **Create repository**。

### 第二步：生成访问令牌（只做一次，代替密码用）

GitHub 推送时**不接受账号密码**，需要一把"钥匙"（token）：

1. github.com 右上角头像 → **Settings**；
2. 左侧最下面 **Developer settings** → **Personal access tokens** →
   **Tokens (classic)**；
3. 点 **Generate new token (classic)**；
4. Note 随便填（如 `kcgtest1-push`），Expiration 选 90 天或 No expiration；
5. 勾选 **repo** 这一整组权限，其他不勾；
6. 拉到页面最下面点 **Generate token**；
7. 立刻复制那串 `ghp_` 开头的字符（**只显示这一次，关掉页面就看不到了**）。

### 第三步：首次推送（只做一次）

```bash
cd ~/WorkPlace/kcgtest1
git push -u origin master
# 会弹出两行提问：
#   Username: 输入 dongtian12138
#   Password: 粘贴刚才复制的 ghp_ 开头的令牌（不是你的登录密码！）
```

输完这一次，凭证已自动保存（`~/.git-credentials`，注意它是明文文件，
别把电脑借给别人），以后 `git push` 不再要密码。

### 第四步：以后的两条命令

```bash
git push      # 这台电脑改完了，推上去（GitHub 是"云端存档本"）
git pull      # 在另一台电脑上，把最新改动拉下来
```

**两台电脑轮流用的铁律**：开干前先 `git pull`，干完了 `git push`。
顺序反了会提示冲突，先把 `git status` 的输出发给我/agent 再操作，别自己乱按。

> 提示：国内网络访问 GitHub 偶尔会慢或连不上，push/pull 失败时可以重试几次；
> 如果一直不行，可以换成 Gitee（操作完全相同，只是网址不同），到时候找我改。

---

## 五、没有网络、也不想用 GitHub？用增量包

每次换机前在旧电脑上打包"上次以来的增量"，拷个小文件过去：

```bash
# 旧电脑（打包增量，假设上次同步点是 v20260814-backup）
cd ~/WorkPlace/kcgtest1
git bundle create ~/kcgtest1_updates_20260901.bundle master ^v20260814-backup

# 新电脑（拉入增量）
cd ~/WorkPlace/kcgtest1
git pull ~/kcgtest1_updates_20260901.bundle master
```

增量包通常只有几十 KB 到几 MB，比拷整个文件夹快得多。

---

## 六、常见报错与对策

| 报错/现象 | 对策 |
| --- | --- |
| Please tell me who you are | 配置身份：`git config user.name "你的名字"` 和 `git config user.email "你的邮箱"` |
| push 被拒绝（rejected） | 先 `git pull` 再 `git push`；还不行就把输出发给我 |
| 403 / Authentication failed | 令牌过期或权限没勾 repo；重新生成一个令牌再 push |
| 忘了 add 就 commit 了 | `git add . && git commit --amend --no-edit` |
| 某个文件改乱了想撤销 | `git checkout -- 文件名`（会丢掉这个文件的未存档改动！） |
| 想看某次存档里的完整状态 | `git checkout <提交号>` —— **初学者先别用**，问 agent |

---

## 七、三条保命规则

1. **只用** 第二、四、五节的命令；不要碰 `reset --hard`、`rebase`、
   `push -f`（会毁掉存档历史）。
2. artifacts、模型权重、抓图**不经过 git**（已被 .gitignore 排除），
   大文件证据继续走 U 盘/移动硬盘。
3. 每次"某个里程碑验收通过"后：`git add . && git commit -m "milestone: ..."`
   并打一个 tag——这就是你项目自己的"冻结基线"。
