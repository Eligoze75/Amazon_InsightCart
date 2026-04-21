# Milestone 2 — RAG answer quality (Claude Haiku)

Discussion of the optional grounded LLM reply in the Streamlit app (`generate_answer` in `src/query_expansion.py`). Retrieval and re-ranking are fixed; this milestone focuses on the **final answer** step.

---

## Why Claude Haiku for the RAG reply

In short, we choose **Claude** because is our prefered and most used LLM in the day to day. We choose **Haiku** because is Anthropic's fastest and most cost effective model. For an explorative academic project, haiku is the best suited for the job.

---

## System prompts compared

We compare the **baseline** shipped in code against one **alternative** for a short ablation.

**Baseline (current, `RAG_SYSTEM_PROMPT`):**

```markdown
You are a helpful Amazon Video Games product assistant. Answer the user's question using only the product information provided. Be concise and specific.
```

**Alternative (shorter, stricter refusal, with no role context):**

```markdown
You answer only from the retrieved product snippets. If they are not enough, say you cannot tell. Two sentences max. No bullet lists.
```

### Side-by-side impressions


| Query | Baseline answer (summary) | Alternative answer (summary) | Preference |
|---|---|---|---|
|Nostalgic videogames for xbox|For nostalgic Xbox gaming, I'd recommend Atari Flashback Classics Vol. 1 (rated 4.4/5), which features 50 classic Atari and arcade titles remastered for Xbox One. If you prefer original Xbox, Atari Anthology (rated 3.7/5) offers 18 arcade hits including classics like Centipede and Asteroids, plus Atari 2600 console games.|For nostalgic Xbox gaming, I'd recommend Atari Flashback Classics Vol. 1, which includes 50 classic Atari and arcade titles remastered for Xbox One, or Atari Anthology for the original Xbox with 18 arcade hits like Centipede and Asteroids. Both offer authentic retro gaming experiences from the golden age of arcade and early console gaming.|I do like the baseline answer better. The change is subtle, but you can notice how the LLM is taking the role of the product assistant and gives you extra information, guides you more. Whereas the the alternative is dry, very focused on only recommending what it sees.|
|Games similar to Sekiro|Based on the retrieved products, I can see that Sekiro: Shadows Die Twice is available, but the search results don't show other similar games. However, the product descriptions mention that Sekiro is from FromSoftware, the creators of Bloodborne and the Dark Souls series — these would be excellent alternatives if you're looking for games with similar challenging action-adventure gameplay and RPG elements.|I cannot provide recommendations for games similar to Sekiro based on these results, as the retrieved products only show different versions of Sekiro itself rather than similar games. To find similar titles, you might want to search for games from the same developers (FromSoftware, creators of Bloodborne and Dark Souls series) or search for "action-adventure games with RPG elements."|Again, I do prefer the baseline better here. This was a case in which the difference is very visible: the baseline makes suggestions with what it has at hand and using the descriptions it offers more. The alternative is very strict and asks the user to use a better query.|

### What differed in practice

Implementing a role in the system prompt really makes the difference. The model behaves differently and it's whole objective seems to change.

I still feel like the model shouldn't mention that it only has access to what was retrieved. We'll make an improvement to improve this behaviour.

---

## Query set for manual evaluation (5 queries)

| # | Query | Intent (why we included it) |
|---|---|---|
| 1 | `wireless PS5 controller` | Easy: clear product type; checks factual grounding. |
| 2 | `best god of war` | Medium: franchise name; checks that the model does not hallucinate titles. |
| 3 | `party game for Switch` | Medium: category + platform; multiple valid products. |
| 4 | `RPG under $30` | Hard: price constraint may be weak in snippets; checks honesty / limits. |
| 5 | `PS5 controller drift` | Hard: opinion / issue; checks whether the answer stays in retrieved text. |

---

## Results (5 queries)

### 1. `wireless PS5 controller`

**Retrieval snapshot (titles or notes):**

- Wireless Keyboard for PS5 Controller, arVin Rechargeable Bluetooth Keypad Gamepad Online Gaming Live Chat Message Chatpad Keyboard with Built-in Speaker & 3.5MM Audio Jack for Playstation 5 Dualsense
- Custom UN-MODDED Wireless PRO Controller compatible with PS5 DualSense Exclusive Unique Design (Mask)
- SADES PS5 Controller Charging Station Wireless Charging Receiver Adapter Compatible With PS5 Controller Gamepad,Fast Charger PS5 Wireless Handle Double Seat Charge

**Answer:**

*I found a couple of options for you! If you're looking for an actual PS5 controller, there's a Custom UN-MODDED Wireless PRO Controller compatible with PS5 DualSense that has an excellent rating of 4.6/5 stars. Alternatively, if you need charging solutions for your PS5 controller, the SADES PS5 Controller Charging Station offers wireless charging capability with a double seat design for convenient charging.*

**Evaluation:**

Yup, perfect answer. This was exactly what I would've expect. The retrieval had just one issue: the ordering is not the best (*Wireless Keyboard for PS5 Controller* is the first result).

---

### 2. `best god of war`

**Retrieval snapshot (titles or notes):**

- God of War: Collection Full Game - PS Vita [Digital Code]
- God of War 3 Favoritos - Spanish/English Edition - PlayStation 3
- God of War 2 - PlayStation 2

**Answer:**

*Based on the available options, God of War 2 for PlayStation 2 stands out as an excellent choice with a 4.5/5 rating—the highest among these titles. It continues Kratos's epic saga as he sits atop the Olympus throne and battles untold horrors across the far reaches of the earth. If you're looking for a complete collection experience, the God of War: Collection for PS Vita offers both the original God of War and God of War II in one package, bringing the classic adventures of ancient Greece to a portable system.*

**Evaluation:**

With what the model had at hand I really think that it gave a great answer. And in this case, the retrieved products make sense. I think maybe retrieving more products would be better.

---

### 3. `party game for Switch`

**Retrieval snapshot (titles or notes):**

- Vegas Party (Nintendo Switch)
- Sports Party - Nintendo Switch Standard Edition
- 30 In 1 Game Collection Vol 1 (Nintendo Switch)

**Answer:**

*Great choice! For party games on Switch, I'd recommend Sports Party (rated 4.3/5), which lets you challenge friends and family to fun sports like Frisbee, basketball, golf, and beach tennis. If you want more variety, the 30 In 1 Game Collection Vol 1 (also 4.3/5) offers 30 different pick-up-and-play games perfect for groups, including classics and skill-based challenges. Both are excellent for multiplayer fun!*

**Evaluation:**

100% Perfect answer and perfect retrieval. The answer is cool, explains and sells the retrived products which also make sense and look awesome.

---

### 4. `RPG under $30`

**Retrieval snapshot (titles or notes):**

- Dark Cloud (Renewed) ($30)
- Codex: Death Guard (Hb) (English) ($40)
- Advanced Dungeons & Dragons: Masterpiece Collection

**Answer:**

*Based on the products available, I'd recommend Dark Cloud (Renewed) for an RPG under $30. This PlayStation 2 game combines RPG action with real-time strategy and features an innovative Georama system that lets you build customized worlds. It has excellent ratings (4.1/5) and offers imaginative, vibrant gameplay at an affordable price point.*

**Evaluation:**

Good answer, the model didn't hallucinate or selled whatever, instead suggested the product that made sense in a inviting way.

---

### 5. `PS5 controller drift`

**Retrieval snapshot (titles or notes):**

- 1 Pair Controller Analog Stick Drift Fix Mod for Xbox One PS4 PS5 Switch Pro Gamepad Game Handle Joystick Drift Repair Board (Black)
- 2 x Analog Stick Drift Fix Mod Handle Rocker Adjusting Plate Game Handle Joystick Drift Repair Board for PS4 PS5 Xbox One Xbox Series X S Controller (Black)
- 10pcs Gamepad Analog Stick Drift Fix Mod, Universal 3D Analog Joystick Drift Fix Mod, Black Easy Using Gamepad Joystick Drift Repair Board, for Game Controllers

**Answer:**

*I found several great options to fix PS5 controller drift! You can choose from analog stick drift fix mod boards that are compatible with PS5 controllers. The most highly-rated option is the 10pcs Gamepad Analog Stick Drift Fix Mod (4.5/5 stars) by SHYEKYO, which offers universal compatibility and easy installation. Alternatively, there are 2-piece repair kits like those from LightFairy and Limentea (both 3.3/5 stars) that include replacement joystick sticks and adjustment plates to correct the drift issue.*

**Evaluation:**

So, it's an interesting answer. Claude understood the query in a very good way. The query is related to issues with a product, and Claude offered products that help with that issue. I think Claude might be performing as an actual salesman.

---

## Conclusions

In short, looks like our pipeline is doing really good. The results make sense, Claude is doing a great job and the system prompt was very helpful.

The retrievers also do a good job in general, but even when something is off, Claude is able to compensate with an answer that makes sense.
