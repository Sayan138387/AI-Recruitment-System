import json


# Load knowledge base
with open(
    "datasets/job_skill_knowledge.json",
    "r",
    encoding="utf-8"
) as file:

    knowledge_base = json.load(file)


def get_required_skills(role):

    role_data = knowledge_base.get(role)

    if role_data is None:
        return []

    return role_data["required_skills"]


def get_related_roles(role):

    role_data = knowledge_base.get(role)

    if role_data is None:
        return []

    return role_data["related_roles"]


def find_matching_skills(resume_text, role):

    required_skills = get_required_skills(role)

    resume_text = resume_text.lower()

    matched_skills = []

    for skill in required_skills:

        if skill.lower() in resume_text:
            matched_skills.append(skill)

    return matched_skills


if __name__ == "__main__":

    role = "Data Scientist"

    print("\nRole:")
    print(role)

    print("\nRequired Skills:")
    print(get_required_skills(role))

    print("\nRelated Roles:")
    print(get_related_roles(role))

    sample_resume = """
    I have experience in Python, SQL, machine learning,
    pandas and statistics.
    """

    print("\nMatched Skills:")
    print(
        find_matching_skills(
            sample_resume,
            role
        )
    )