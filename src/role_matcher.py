import json


with open(
    "datasets/job_skill_knowledge.json",
    "r",
    encoding="utf-8"
) as file:
    knowledge_base = json.load(file)


def find_best_role(resume_text):

    resume_text = resume_text.lower()

    role_scores = []

    for role, data in knowledge_base.items():

        required_skills = data["required_skills"]

        matched_skills = []

        for skill in required_skills:

            if skill.lower() in resume_text:
                matched_skills.append(skill)

        if required_skills:
            score = (
                len(matched_skills) /
                len(required_skills)
            ) * 100
        else:
            score = 0

        role_scores.append(
            {
                "role": role,
                "score": score,
                "matched_skills": matched_skills
            }
        )

    role_scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return role_scores