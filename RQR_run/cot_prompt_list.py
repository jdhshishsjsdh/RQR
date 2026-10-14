STAGE1_VISUAL_FACTS = """Please carefully analyze this image and extract all structured information.

1. Entities: List all visible independent objects/persons/animals. Use simple and unique labels to identify them.
   - If there are multiple similar objects, use suffixes like "person_left", "person_right"

2. Attributes (unary predicates): For each entity, what properties does it have?
   Use the following predicates:
   - Color: IsRed, IsBlue, IsGreen, IsYellow, IsWhite, IsBlack, IsBrown, IsOrange, IsPink, IsPurple, IsGray, IsBeige
   - Size: IsLarge, IsSmall, IsTall, IsShort, IsWide, IsNarrow
   - State: IsOpen, IsClosed, IsMoving, IsStatic, IsStanding, IsSitting, IsLying, IsWalking, IsRunning, IsSmiling
   - Material: IsWooden, IsMetal, IsPlastic, IsGlass, IsFabric, IsLeather
   - Other: IsOld, IsNew, IsShiny, IsBright, IsDark, IsWet, IsTransparent

3. Relationships (binary predicates):
   Use predicates such as:
   - Spatial: On, Under, Behind, InFrontOf, LeftOf, RightOf, Inside, NextTo, Above, Below, Near, Contains
   - Actions: Holding, Wearing, Riding, Eating, Carrying, Pushing, Touching
   - Associations: PartOf, BelongsTo, Supports, Using

4. Negated facts: Things that are clearly not true.
   Important for detecting false premises.
   Only include high-confidence negations.

Key: Only state what you can clearly see.

Output using only JSON format:
{
  "entities": ["entity1", "entity2"],
  "unary_facts": [
    {"predicate": "IsRed", "entity": "car"}
  ],
  "binary_facts": [
    {"predicate": "On", "subject": "cat", "object": "table"}
  ],
  "negated_facts": [
    {"predicate": "IsBlue", "entity": "car", "type": "unary"}
  ]
}
"""

STAGE2_REASONING = """You are a fake news detection system. Given a news text and an image, determine whether the news is REAL or FAKE.

---
The news content is [{text}]. The following information has been extracted from the image:
Entities: 
{entities}
Facts: 
{facts}

---
Based on the provided content, please reason and identify the veracity of the news by following these steps:

1. First, retrieve general knowledge about this type of news and the formal content style used for creating such news, and use them to verify the truthfulness of the conclusions. If you can clearly determine a result of real or fake, proceed directly to Step 3. Otherwise, continue to Step 2.

2. If a knowledge-oriented judgment cannot be reached, perform a content-oriented analysis following these sub-steps until a clear conclusion is obtained:
(1) Distinguish between original shot footage and creator-added content. Identify and separate entities that actually exist in the image from elements added during post-production.
(2) Identify creator-added text details, including the specific text content, color, and screen position (e.g., top-left, center, bottom-right).
(3) Identify entities that are critical to authenticity assessment, including but not limited to: entity objects and facts consistent with the original text description, forged official announcements or notices, and other relevant visual evidence.
(4) Conduct spatial localization and detailed analysis of key entities. Examine their positions in the image and extract clues that help verify the authenticity of the news content.
(5) Identify all individuals appearing in the image and their identities/roles, such as: parties involved in the incident, law enforcement officers, victims, medical personnel, and other relevant persons.

3. Synthesize the results from Steps 1 and 2, combined with your professional knowledge, to reason step by step and determine the veracity of the news. The final output must be in JSON format, containing the following fields:
- "final_answer": "REAL" or "FAKE"
- "reasons": A list of brief reasons

---
## Example (EXAMPLE ONLY):

News Text:
According to an anonymous source, on January 15, 2024, a large-scale protest took place at Tiananmen Square in Beijing, and more than 10,000 people have died in the clashes. The same report also stated that order was maintained at the scene and no serious clashes occurred. According to an anonymous source, a large-scale protest took place at Tiananmen Square in Beijing on January 15, 2024, and more than 10,000 people have died in the clashes. The same report also stated that order was maintained at the scene and no serious clashes occurred.

Output:
{{
  "final_answer": "FAKE",
  "confidence": 0.95,
  "reasons": [
    "Internal contradiction: claims both conflict and good order",
    "Numerical anomaly: 10000 deaths without official confirmation",
    "Unreliable source: anonymous source cited"
  ]
}}

---
## IMPORTANT:
1. Output ONLY the JSON object. Start with {{ and end with }}
2. Do NOT include any explanation before or after the JSON

## Now start output:
"""


def flatten_facts(visual_facts):
    entities = ", ".join(str(e) for e in visual_facts["entities"])
    lines = []
    for item in visual_facts["unary_facts"]:
        lines.append("%s(%s)" % (item["predicate"], item["entity"]))
    for item in visual_facts["binary_facts"]:
        lines.append("%s(%s, %s)" % (item["predicate"], item["subject"], item["object"]))
    for item in visual_facts["negated_facts"]:
        lines.append("NOT %s(%s)" % (item["predicate"], item["entity"]))
    return entities, ";\n".join(lines)


def build_stage2_prompt(visual_facts, news_text):
    entities, facts = flatten_facts(visual_facts)
    return STAGE2_REASONING.format(text=news_text, entities=entities, facts=facts)
