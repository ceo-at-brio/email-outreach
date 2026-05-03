"""
Verbatim instruction blocks from Email - Prompts.pdf (for Gemini generation).
"""

# --- Intro Email (page 1) ---
INTRO_EMAIL_INSTRUCTIONS = r'''Intro Email — first outreach message prompt:

"I am Parag, Founder of SoftwareBrio. Write a warm, expert-led outreach email (under 350 characters total) using this structure: Subject Line: A short 'Google Hook' (e.g., 'Google-scale systems for [Company Name]'). First Para: Start with a 1-sentence genuine compliment. Then, use this specific logic: 'At Google, I saw how fast growth usually leads to [Headache 1] and [Headache 2]. Those are exactly the kinds of technical hurdles we solve at SoftwareBrio.' (Choose headaches relevant to the client's industry). Second Para: Offer a free tech proposal to simplify their operations and get their systems talking to each other. Closing: Ask for a brief intro meeting in the coming days.

CHARACTER LIMIT : 350
Constraints: * Keep the total message concise and professional * Preserve meaning; do not over-shorten * No sales pitch, no CTA, no links * Use calm, intelligent language * No words like "curious", "excited", or "keen" * The message must feel peer-to-peer, not outreach-heavy

Generate only the final message.

sample : Hi Joey, your community-first approach to scaling Fitness Collective is impressive. At Google, I saw how rapid growth often leads to redundant software and rising cloud costs — challenges we help solve at SoftwareBrio. Open to a brief intro next week? Happy to share a quick, free tech proposal. Parag | Linkedin Founder, SoftwareBrio.com Built by ex-Meta & Google engineers"
'''

# --- First followup (page 1) ---
FIRST_FOLLOWUP_INSTRUCTIONS = r'''First follow up:

Write an email follow-up message strictly under 300 characters for someone who hasn’t replied to my introductory mail. so i want you to mail them without praising them and Start with Hi Name Following up on my previous mail. Share a unique insight, tip, or resource relevant to their post, hiring, or business challenge. Briefly mention how we can help with a solution, talent, or service. then ask do you or your ventures outsource AI or Software development work? And at last also ask for a call to explore potential synergies. and then sign off with : Parag | Linkedin Founder, SoftwareBrio.com Built by ex-Meta & Google engineers Also you know how email format works and what mail sending trend is so try to write it in that format which is so engaging and catches the eye because most of the people will open it in mobile phone so it should look classy and short so they will read it. Message length must be strictly under 300 chars.

Sample: Hi Benjamin, Following up on my previous mail. Many healthcare tax firms face CRM and reporting tool disconnects that slow compliance visibility. We can help you unify systems securely. Do you or your ventures outsource AI or software development work? Open to a short call? Parag | Linkedin Founder, SoftwareBrio.com Built by ex-Meta & Google engineers
'''

# --- Second followup (page 2) ---
SECOND_FOLLOWUP_INSTRUCTIONS = r'''Second follow up:

For the second follow up mail, what you have to do is prepare a engaging follow up to them so that they reply because they have not replied the previous two mails so i want you to make a that kind of mail which will force them to reply, the msg should be that good. and ask them for a call also so follow up previous mails and take idea from his profile as well as from the previous mails and prep best follow up mail by saying how we could actually help them and also ask for a call and start with greeting and then say Following up on my previous mail! and then tell how softwarebrio can actually help them and then ask for a call also check previous messages which we have send and say according to that follow up those messages up dont write same things follow up those messages so it looks like you are there noticing and also it will look like we are actually present and it is not ai spamming And at last don't forget to add Best, Parag, Google | Founder, SoftwareBrio.com Message length should be strictly less than 260 chars

Understand? Start with a greeting. Say: "Hope you're doing well!" Refer to their profile & previous message context. Show how SoftwareBrio can help them in a specific way. Ask for a call. End with: Best, Parag, Google | Founder, SoftwareBrio.com" Keep it <270 characters. PLEASE BE SPECIFIC ABOUT THE WORK BY TAKING IDEA FROM THEIR PROFILE, RECENT ACTICITIES AND COMPANY'S WORK you have to focus on things like the recent activities of the person’s profile and our previous mails and take the best out of that and use it in a message got it?

Sample : Message: Hi Taro, We’ve analyzed engagement patterns for creator-led Web3 platforms and spotted 3 AI-driven ways RACER & KURUMA could boost fan interaction and NFT monetization immediately. I’d love to share a custom blueprint in a 15-min call—no strings attached. Best, Parag, Google | Founder, SoftwareBrio.com"
'''
