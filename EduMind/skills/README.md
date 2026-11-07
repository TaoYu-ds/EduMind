# EduMind Skills 文档

EduMind 启动时会从 `EDUMIND_SKILLS_DIR` 读取 Skills，并在匹配学生请求时注入到对应 Agent 的 system prompt。Skills 适合维护教学规范、计划编排规则、考试合规边界、分流话术和禁止事项。

当前内置四类 Skills（教育培训场景）：

```text
skills/general_learning_service/SKILL.md  # 学习服务台：接待、分流、平台使用、调课退费、情绪风险
skills/tutor_guidance/SKILL.md            # 学科辅导：卡点诊断、分层讲解、作业辅导边界
skills/study_plan/SKILL.md                # 学习计划：时间编排、负荷校验、课程推荐边界
skills/exam_service/SKILL.md              # 考试服务：报名/准考证/成绩流程与合规约束
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill_name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: 学科辅导教学规范
description: 适用于 TutorAgent 的知识点讲解与作业辅导规范
keywords: 不会,不懂,这道题,作业,公式,原理,步骤
agents: tutor
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称，会出现在注入给模型的 prompt 中。
- `description`：简短说明，方便 `/skills` 接口排查。
- `keywords`：触发关键词，学生消息命中后才注入；多个关键词用英文逗号或中文逗号分隔均可。
- `agents`：适用 Agent，可填 `general`、`tutor`、`plan`、`exam`，多个值用逗号分隔。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 重要规则放在文档前半部分，因为过长内容会按 prompt 预算截断。
- 一类 Skill 只描述一类职责，不要把讲题、排计划、考试政策混在一个文件里。
- 必须包含“角色定位”“处理流程”“边界/禁止事项”等稳定章节。
- 教学类 Skill 要写明“不代写作业、不编造教材原文”；考试类 Skill 要写明“不编造时间与分数线”。
- 对学生隐私、身份证号、准考证号、银行卡号、密码、验证码必须写明禁止收集。
- 对无法保证的事项使用保守措辞，例如“通常”“以官方公告为准”“需要核验后确认”。
- 对需要人工、教师、心理支持的场景要明确写出升级条件。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看加载结果和解析错误：

```bash
curl http://localhost:8000/skills
```
