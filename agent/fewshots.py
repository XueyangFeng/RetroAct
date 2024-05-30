WEBTHINK_SIMPLE6 = """
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: Finish[yes]"""

REFLECTIONS = """
Previous Trial:
Question: Kam Heskin plays Paige Morgan in a 2004 film directed by who?
Thought 1: I need to search Kam Heskin and Paige Morgan, find the 2004 film they are in, then find the director of the film.
Action 1: Search[Kam Heskin]
Observation 1: Kam Erika Heskin (born May 8, 1973) is an American actress. She began her career playing Caitlin Deschanel on the NBC daytime soap opera Sunset Beach (1998–1999), before appearing in films Planet of the Apes (2001) and Catch Me If You Can (2002). Heskin went on in 2003 to play Elizabeth Bennet in an independent film Pride & Prejudice: A Latter-Day Comedy, Robin in Charmed (2004),  and Paige Morgan in The Prince and Me film franchise (2006–2010).
Thought 2: Kam Heskin plays Paige Morgan in The Prince and Me film franchise. I need to search The Prince and Me film franchise and find the director of the 2004 film.
Action 2: Search[The Prince and Me film franchise]
Observation 2: Could not find [The Prince and Me film franchise]. Similar: ['Kam Heskin', 'Prince of Persia', 'Harry Potter and the Half-Blood Prince (film)', 'Prince of Persia: The Sands of Time (film)', 'The Chronicles of Narnia (film series)', 'Majestic Prince (franchise)', 'Kingsman (franchise)', 'The Little Prince (2015 film)', 'Shrek (franchise)', 'Harry Potter (film series)']
Thought 3: I need to search The Prince and Me (2004 film) instead.
Action 3: Search[The Prince and Me (2004 film)]
Observation 3: Could not find [The Prince and Me (2004 film)]. Similar: ['The Prince & Me', 'The Prince & Me 2: The Royal Wedding', 'Kam Heskin', 'Prince of Darkness (film)', 'Prince of Persia: The Sands of Time (film)', 'Rob Knox', 'Alexander (2004 film)', 'Prince of Persia', 'Brooklynn Prince', 'Prince (musician)']
Thought 4: I need to search The Prince & Me (2004 film).
Action 4: Search[The Prince & Me (2004 film)]
Observation 4: Could not find [The Prince & Me (2004 film)]. Similar: ['The Prince & Me', 'The Prince & Me 2: The Royal Wedding', 'Prince of Darkness (film)', 'Prince of Persia: The Sands of Time (film)', 'Rob Knox', 'Alexander (2004 film)', 'Prince (musician)', 'Prince of Persia', 'Kam Heskin', 'Brooklynn Prince']
Thought 5: I need to search The Prince & Me (2004 film) instead.
Action 5: Search[The Prince & Me (2004 film)]
Observation 5: Could not find [The Prince & Me (2004 film)]. Similar: ['The Prince & Me', 'The Prince & Me 2: The Royal Wedding', 'Prince of Darkness (film)', 'Prince of Persia: The Sands of Time (film)', 'Rob Knox', 'Alexander (2004 film)', 'Prince (musician)', 'Prince of Persia', 'Kam Heskin', 'Brooklynn Prince']
Thought 6: I need to search The Prince & Me (2004 film).
Action 6: Search[The Prince & Me (2004 film)]
Observation 6: Could not find [The Prince & Me (2004 film)]. Similar: ['The Prince & Me', 'The Prince & Me 2: The Royal Wedding', 'Prince of Darkness (film)', 'Prince of Persia: The Sands of Time (film)', 'Rob Knox', 'Alexander (2004 film)', 'Prince (musician)', 'Prince of Persia', 'Kam Heskin', 'Brooklynn Prince']

Reflection: I got stuck in a loop where I kept trying to search 'The Prince & Me (2004 film)' but the page could not be found. Instead I should have tried to search the similar results that had a similar name to see and they were made in 2004.
"""
COTQA_SIMPLE6 = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: Finish[Richard Nixon]

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action: Finish[The Saimaa Gesture]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action: Finish[Arthur's Magazine]

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action: Finish[Yes]"""

COT_SIMPLE_REFLECTION = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the Rocky Mountains. The Rocky Mountains rise in elevation from around 1,800 to 14,000 ft, so the answer is 1,800 to 14,000 ft.
Action: Finish[1,800 to 14,000 ft]

Reflection: My reasoning for the elevation range of the eastern sector of the Colorado orogeny failed because I incorrectly associated it with the Rocky Mountains instead of the High Plains. In the future, when attempting this question, I should focus on better understanding the geography and geological formations involved in the Colorado orogeny to avoid confusion.

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after a prominent cartoon character, Mickey Mouse, so the answer is Mickey Mouse.
Action: Finish[Mickey Mouse]

Reflection: My reasoning for the naming of the character Milhouse in "The Simpsons" failed because I incorrectly assumed that the character was named after a prominent cartoon character, Mickey Mouse. In the future, when attempting this question, I should focus on better researching the background of "The Simpsons" and Matt Groening's influences in naming characters to avoid making incorrect assumptions. 
"""

COT = """Relevant Context: The Nile River is the longest river in the world, spanning approximately 6,650 kilometers (4,132 miles) in length. It flows through eleven countries in northeastern Africa, including Egypt, Sudan, and Uganda.
Question: What is the longest river in the world?
Thought: The question asks for the longest river in the world, which I know is the Nile River based on the context provided.
Action: Finish[Nile River]

Relevant Context: Ludwig van Beethoven was a German composer and pianist who was a crucial figure in the transition between the Classical and Romantic eras in Western classical music. One of his most famous compositions is the Symphony No. 9, also known as the "Choral" symphony.
Question: Which composer created Symphony No. 9?
Thought: The question is asking for the composer of Symphony No. 9. Based on the context, I know that Ludwig van Beethoven composed this symphony.
Action: Finish[Ludwig van Beethoven]

Relevant Context: Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy. During this process, plants absorb sunlight, carbon dioxide, and water to produce glucose and oxygen.
Question: What do plants produce during photosynthesis?
Thought: The question is asking about the products of photosynthesis. From the context, I know that plants produce glucose and oxygen during this process.
Action: Finish[Glucose and Oxygen]
"""

COT_REFLECT = """
Relevant Context: Ernest Hemingway's novel "The Old Man and the Sea" tells the story of Santiago, an aging Cuban fisherman, who struggles to catch a giant marlin in the Gulf Stream. The book won the Pulitzer Prize for Fiction in 1953 and contributed to Hemingway's Nobel Prize for Literature in 1954.
Question: Which literary award did "The Old Man and the Sea" contribute to Hemingway winning?
Thought: The question is asking which award "The Old Man and the Sea" contributed to Hemingway winning. Based on the context, I know the novel won the Pulitzer Prize for Fiction and contributed to his Nobel Prize for Literature.
Action: Finish[Pulitzer Prize for Fiction]

Reflection: My answer was correct based on the context, but may not be the exact answer stored by the grading environment. Next time, I should try to provide a less verbose answer like "Pulitzer Prize" or "Nobel Prize."

Context: On 14 October 1947, Chuck Yeager, a United States Air Force test pilot, became the first person to break the sound barrier by flying the Bell X-1 experimental aircraft at an altitude of 45,000 feet.
Charles Elwood "Chuck" Yeager (13 February 1923 - 7 December 2020) was a United States Air Force officer, flying ace, and test pilot. He is best known for becoming the first person to break the sound barrier, which he achieved in the Bell X-1 aircraft named Glamorous Glennis. Yeager was also a distinguished fighter pilot during World War II and was credited with shooting down at least 12 enemy aircraft. In 1973, he was inducted into the National Aviation Hall of Fame for his significant contributions to aviation.
Question: Who is the first person to break the sound barrier?
Thought: The question is asking for the first person to break the sound barrier. From the context, I know that Chuck Yeager, a United States Air Force test pilot, was the first person to break the sound barrier.
Action: Finish[Chuck Yeager]

Reflection: Upon reflecting on the incorrect answer I provided, I realize that I may not have given the full name of the individual in question. In the context, both the given name and the nickname were mentioned, and I only used the nickname in my response. This could have been the reason my answer was deemed incorrect. Moving forward, when attempting this question again or similar questions, I will make sure to include the complete name of the person, which consists of their given name, any middle names, and their nickname (if applicable). This will help ensure that my answer is more accurate and comprehensive."""

COT_REFLECT2 = """Relevant Context: The novel "To Kill a Mockingbird" was written by Harper Lee and published in 1960. The story takes place in the fictional town of Maycomb, Alabama during the Great Depression. The main characters are Scout Finch, her brother Jem, and their father Atticus Finch, a lawyer.
Question: Where does "To Kill a Mockingbird" take place?
Thought: The question is asking for the setting of "To Kill a Mockingbird." Based on the context, I know that the story takes place in the fictional town of Maycomb, Alabama.
Action: Finish[Mississippi]

Reflection: I made an error in my response, as I incorrectly stated that "To Kill a Mockingbird" takes place in Mississippi. Upon reviewing the context, I realized that the correct answer is the fictional town of Maycomb, Alabama. I may have been confused due to the story's Southern setting. Next time, I should be more cautious and double-check the context before providing an answer.

Relevant Context: Sir Isaac Newton formulated the three laws of motion that are fundamental to classical mechanics. These laws describe the relationship between the motion of an object and the forces acting upon it. They are known as Newton's First Law, Newton's Second Law, and Newton's Third Law.
Question: Who formulated the three laws of motion?
Thought: The question is asking for the person who formulated the three laws of motion. Based on the context, I know that Sir Isaac Newton formulated these laws.
Action: Finish[Albert Einstein]

Reflection: My response was incorrect, as I stated that Albert Einstein formulated the three laws of motion, when the correct answer is Sir Isaac Newton. I may have confused the two scientists due to their significant contributions to the field of physics. In the future, I should be more careful in distinguishing between different scientists and their specific accomplishments before answering a question.

Relevant Context: The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was built primarily to protect the Chinese Empire from invasions by various nomadic groups. The wall stretches over 13,000 miles (21,196 kilometers) and was constructed by several Chinese dynasties.
Question: What is the primary purpose of the Great Wall of China?
Thought: The question is asking about the primary purpose of the Great Wall of China. From the context, I know that it was built primarily to protect the Chinese Empire from invasions by various nomadic groups.
Action: Finish[Trade]

Reflection: My answer was incorrect, as I stated that the primary purpose of the Great Wall of China was for trade. After reevaluating the context, I realized that the wall was built primarily to protect the Chinese Empire from invasions by various nomadic groups. My confusion may have stemmed from the fact that the wall is a significant historical landmark that may have played a role in trade routes. Next time, I will focus on the key details provided in the context to ensure the accuracy of my response.
"""