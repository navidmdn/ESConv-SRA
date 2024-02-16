from typing import List, Literal, TypedDict


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

strategy_descriptions = {
    "Question": """In emotional support counseling, asking open-ended questions is crucial. These questions encourage the\
 individual to express their thoughts and feelings. They can be used to explore the person's emotions, triggers,\
 or concerns. For instance, you might ask, "What is causing you to feel this way?" """,
    "Restatement or Paraphrasing": """This technique involves repeating or rephrasing what the person has said to show\
 that you're actively listening and trying to understand. For instance, if someone says, "I'm really stressed about\
 work," you can respond with, "It sounds like work is causing you a lot of stress." """,
    "Reflection of Feelings": """Reflecting feelings involves acknowledging and validating the person's emotions. This\
 helps the individual feel heard and understood. If someone expresses sadness, you might respond with, "I can see that\
 you're feeling very sad about this situation." """,
    "Self-disclosure": """Carefully sharing your own relevant experiences or feelings can help build rapport and trust.\
 However, it should be done in a way that doesn't shift the focus from the person seeking support. For example,\
 "I've also felt overwhelmed at times, and it's completely normal." """,
    "Affirmation and Reassurance": """Providing affirmations and reassurance helps boost the individual's self-esteem and\
 confidence. You might say, "You're doing a great job by seeking help, and I believe you have the strength to overcome\
 this." """,
    "Providing Suggestions": """In emotional support counseling, suggesting helpful actions or strategies can guide\
 individuals without imposing decisions. For example, if someone is stressed, you might suggest trying relaxation\
 techniques like deep breathing.""",
    "Information": """Providing relevant facts or resources is crucial in emotional support. Offering accurate information\
 about available therapies, self-help options, or support groups can empower individuals to make informed decisions\
 about their emotional well-being."""
}


modified_extes_support_strategies = {
    "Affirmation": "Description: This involves acknowledging and positively reinforcing an individual's strengths, feelings, or actions. Examples: 'You've shown incredible resilience in facing these challenges.' 'I admire your dedication to improving your situation.' 'Your ability to stay hopeful in tough times is truly commendable.'",
    "Avoid Judgment and Criticism": "Description: This strategy focuses on providing support without expressing negative judgments or criticisms of the person's thoughts, feelings, or actions. Examples: 'It's understandable that you felt that way in that situation.' 'Everyone makes mistakes, and it's okay to be imperfect.' 'Your feelings are valid, and it's okay to express them.'",
    "Clarification": "Description: This entails asking questions or restating what was said to ensure clear understanding of the person's feelings or situation. Examples: 'Could you explain a bit more about what you mean by that?' 'So, what you're saying is that you feel overwhelmed by the workload?' 'I want to make sure I understand; you're feeling anxious about the upcoming event, right?'",
    "Collaborative Planning": "Description: This involves working together to develop strategies or plans to address specific issues or challenges. Examples: 'Let's brainstorm some strategies that could help you manage this stress.' 'We can work together to come up with a plan that feels comfortable for you.' 'How about we outline some steps you can take to approach this problem?'",
    "Emotional Validation": "Description: This strategy involves acknowledging and accepting the person's emotions as legitimate and important. Examples: 'It's completely normal to feel sad in a situation like this.' 'Your feelings of frustration in this case are absolutely understandable.' 'I hear you, and it makes sense that you would feel anxious about this.'",
    "Normalize Experiences": "Description: This approach helps the person understand that their experiences or feelings are common and not something to be ashamed of. Examples: 'Many people go through similar challenges, and it's okay to feel this way.' 'Feeling overwhelmed in such situations is a common reaction.' 'It's normal to have ups and downs in response to life's stresses.'",
    "Offer Hope": "Description: This involves providing reassurance that things can improve and that there is hope for a better future. Examples: 'I'm confident that you'll find a way through this challenge.' 'Things might be tough now, but there is always a possibility for change and growth.' 'I believe in your ability to overcome these obstacles.'",
    "Promote Self-Care Practices": "Description: Encouraging the person to engage in activities that promote physical, emotional, and mental well-being. Examples: 'Have you considered setting aside some time for relaxation or a hobby you enjoy?' 'Taking care of your health is important, maybe try some exercise or meditation.' 'Remember to take breaks and do things that make you feel good.'",
    "Provide Different Perspectives": "Description: Offering new viewpoints or ways of thinking about a situation to help broaden understanding and possibly reduce distress. Examples: 'Have you considered looking at the situation from this angle?' 'Sometimes, stepping back and viewing things differently can be helpful.' 'What if we think about the potential positive outcomes of this scenario?'",
    "Reflective Statements": "Description: Mirroring back what the person has said to show understanding and empathy. Examples: 'It sounds like you're feeling really overwhelmed by your workload.' 'You seem to be saying that this situation has made you feel anxious.' 'I hear that you're finding it hard to cope with these changes.'",
    "Reframe Negative Thoughts": "Description: Helping to shift negative or unhelpful thought patterns into more positive or realistic ones. Examples: 'Instead of thinking of it as a failure, could we see it as a learning opportunity?' 'What if we try to focus on what you can control in this situation?' 'Let's look for the strengths you've shown in dealing with this.'",
    "Share Information": "Description: Providing factual information or resources that might be helpful in understanding or coping with a situation. Examples: 'I read an article about coping strategies that might be useful for you.' 'There are some great books that offer insights into managing these feelings.' 'I can share some websites that provide helpful tips on stress management.'",
    "Stress Management": "Description: Offering techniques or suggestions to help reduce or manage stress. Examples: 'Have you tried deep breathing or mindfulness exercises to manage stress?' 'Creating a regular routine can sometimes help in reducing stress levels.' 'Exercise can be a great way to relieve stress and improve mood.'",
    "Suggest Options": "Description: Presenting various possibilities or alternatives that the person might consider in their situation. Examples: 'One option might be to talk to someone you trust about what you're going through.' 'Have you thought about joining a support group for this issue?' 'Maybe trying a new approach to this problem could yield different results.'",
    "Chit Chat": "Description: Engaging in light, casual conversation to build rapport and provide a sense of normalcy and comfort. Examples: 'How's your day going so far?' 'Did you see that funny movie that came out recently?' 'I love this weather we're having. Do you enjoy outdoor activities?'"
}
