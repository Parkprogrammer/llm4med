STAGE_1_TEMPLATE = """Analyze the following patient information:
- Disease Classification: {disease_classification}
- Related Departments: {departments}
- Affected Systems: {systems}
- Potential Symptoms/Complications: {symptoms_complications}
- General Characteristics: Gender ({gender}), Age ({age}), BMI ({bmi})

Based on this information, summarize the patient's main health issues and areas requiring management."""

STAGE_2_TEMPLATE = """Based on the patient analysis from Stage 1, propose specific management plans for the following areas:
1. Comprehensive disease management
2. Individual management strategies for each potential complication
3. Recommendations for lifestyle improvements
4. Regular check-up and monitoring plan

Please provide 2-3 specific management approaches for each area."""

STAGE_3_TEMPLATE = """Given the available content list:
{content_list}

Considering the patient information analyzed and the management plan proposed, recommend the top 5 most suitable contents for this patient. Provide a brief reason for each recommendation.

[RECOMMEND]
1. [Recommended content 1]: [Reason]
2. [Recommended content 2]: [Reason]
3. [Recommended content 3]: [Reason]
4. [Recommended content 4]: [Reason]
5. [Recommended content 5]: [Reason]
"""