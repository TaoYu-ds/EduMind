"""会话中的学习槽位；只继承已提取的信息，不把槽位齐全视为身份核验。

教育场景的槽位有三类生命周期：
  1. 学生身份类（grade / subject / exam_name）：跨任务复用，聊到别的学科也不该丢；
  2. 学科局部类（chapter / difficulty）：只在同一学科内有效，换学科必须清空，
     否则会把"数学-函数"的薄弱点带到物理提问上；
  3. 任务类（duration / study_time / date）：只在同一任务域内有效。

`merge_state` 负责这套继承规则，避免多轮对话里反复追问已经说过的信息。
"""
import re

# 只属于某一门学科的槽位：换学科即失效。
_SUBJECT_SCOPED = {"chapter", "difficulty"}
# 跨学科、跨任务都保留的学生身份槽位。
# exam_name 是长期目标（高二学生问高考，不会因为换了学科就消失），因此也放在这里。
_STUDENT_SCOPED = {"grade", "subject", "exam_name"}

# 意图分组 → 任务域。同一任务域内槽位可以继承。
_DOMAIN_BY_GROUP = {
    "query": "learning",
    "knowledge_query": "learning",
    "homework_help": "learning",
    "course_recommend": "learning",
    "request": "study",
    "study_plan": "study",
    "exam_info": "exam",
}
_NEUTRAL_DOMAINS = {"greeting", "feedback", "other", "complaint", "escalation"}


def _scope_of(field_name: str) -> str:
    if field_name in _STUDENT_SCOPED:
        return "student"
    if field_name in _SUBJECT_SCOPED:
        return "subject"
    return "task"


def merge_state(previous, entities, intent_group, message, request_id):
    previous = previous if isinstance(previous, dict) else {}
    domain = _DOMAIN_BY_GROUP.get(intent_group, intent_group)
    neutral = domain in _NEUTRAL_DOMAINS or intent_group in _NEUTRAL_DOMAINS
    old = previous.get("entities", {})
    sources = previous.get("sources", {})
    old = old if isinstance(old, dict) else {}
    sources = sources if isinstance(sources, dict) else {}

    old_subject = old.get("subject") or []
    new_subject = entities.get("subject") or []
    changed_subject = bool(new_subject and old_subject and new_subject != old_subject)
    new_task = bool(re.search(r"(?:换个|新的|另一个|别的|另一门|换一门)(?:问题|学科|章节|科目)?", message))

    if neutral:
        # 寒暄/投诉类消息不改变任务域，沿用上一轮；只有明确提到考试或学科时才切换。
        previous_domain = previous.get("domain")
        if new_task:
            domain = "exam" if entities.get("exam_name") else "general"
        elif previous_domain:
            domain = previous_domain
        else:
            domain = "exam" if entities.get("exam_name") else ("learning" if new_subject else "general")

    keep = not new_task and (neutral or domain == previous.get("domain"))

    merged, origins = {}, {}
    for key, value in old.items():
        if not isinstance(value, list):
            continue
        scope = _scope_of(key)
        retain = scope == "student" or (keep and scope in {"subject", "task"})
        if retain:
            merged[key] = list(value)
            if key in sources:
                origins[key] = sources[key]

    # 新学科出现时，上一门学科的章节与难度不再适用。
    if changed_subject:
        for key in _SUBJECT_SCOPED:
            merged.pop(key, None)
            origins.pop(key, None)

    for key, value in entities.items():
        if value:
            merged[key] = list(value)
            origins[key] = request_id
        else:
            merged.setdefault(key, [])
    return {"domain": domain,
            "entities": merged, "sources": origins}
