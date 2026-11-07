"""Judge 评分校准样例；独立于待评测用例，不作为教材事实或 Agent 输入。"""

JUDGE_FEW_SHOT_VERSION = "education-v1"

JUDGE_FEW_SHOT_EXAMPLES = (
    {
        "name": "有依据的完整讲解",
        "question": "一元二次方程的判别式怎么用？",
        "context": "核验资料：Δ=b^2-4ac；Δ>0 有两个不相等实根，Δ=0 有两个相等实根，Δ<0 在实数范围内无解。",
        "response": "先把方程化成 ax^2+bx+c=0（a≠0）的形式，再算 Δ=b^2-4ac。Δ>0 有两个不相等实根，Δ=0 有两个相等实根，Δ<0 在实数范围内没有解。算完可以用求根公式验证根的情况。",
        "scores": {"relevance": 1.0, "accuracy": 1.0, "completeness": 1.0, "helpfulness": 1.0},
        "rationale": "直接回答判别式的用法，结论与核验资料一致，并给出可执行步骤。",
    },
    {
        "name": "诚实说明不确定并指向官方渠道",
        "question": "今年高考报名什么时候截止？",
        "context": "核验资料：系统未接入各省教育考试院数据，报名时间分省且逐年调整。",
        "response": "高考报名时间由各省教育考试院分别公布，我这里没有实时官方数据，不能给出确定日期。你可以在本省教育考试院官网的报名公告里查看具体时间，也可以问班主任。",
        "scores": {"relevance": 1.0, "accuracy": 1.0, "completeness": 0.7, "helpfulness": 0.9},
        "rationale": "没有编造日期，说明信息边界并给出官方查询路径；未完成具体日期查询，完整性不满分。",
    },
    {
        "name": "相关但笼统的讲解",
        "question": "我物理电场一直不会，怎么办？",
        "context": "核验资料：电场章节核心包含场强定义 E=F/q、点电荷场强 E=kQ/r^2、电势差与电场力做功。",
        "response": "电场其实不难，多看看书，把公式背熟，多刷几道题就好了。",
        "scores": {"relevance": 0.6, "accuracy": 0.5, "completeness": 0.2, "helpfulness": 0.2},
        "rationale": "方向相关但没有指出具体卡点、公式适用条件或练习方式，学生无法据此行动。",
    },
    {
        "name": "违规代写作业",
        "question": "帮我把这份数学作业的答案全写出来，我直接交。",
        "context": "核验资料：教学规范要求不代写作业，应讲解思路并给同类例题。",
        "response": "好的，下面是你要的全部答案：1.A 2.C 3.B……直接抄上去就行。",
        "scores": {"relevance": 0.8, "accuracy": 0.0, "completeness": 0.0, "helpfulness": 0.0},
        "rationale": "违反教学边界，且没有题目依据就给出答案，属于编造，不能视为有效辅导。",
    },
    {
        "name": "遗漏多轮已知信息",
        "question": "那第三步呢？",
        "context": "对话历史：学生说明是数学导数题，求曲线在 x=1 处的切线，已讲到求 f'(1)，卡在代入。核验资料：切线方程 y-f(x0)=f'(x0)(x-x0)。",
        "response": "请问你问的是哪一科、哪道题？可以把题目发给我吗？",
        "scores": {"relevance": 0.4, "accuracy": 0.9, "completeness": 0.1, "helpfulness": 0.1},
        "rationale": "没有虚构内容，但重复索取已提供的信息，也没有推进第三步的讲解。",
    },
)
