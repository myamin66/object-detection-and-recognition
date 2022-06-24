1. Mode 1: Detect the direction of the closest object. Eg: If the closest object is in front of the user -> say Front
               VA: Ready -> VA: Front -> The closer the user is, the more it will say
2. Mode 2: Find object.
               VA: What do you want to find?
               User: My phone
                VA: If can't find the object after X seconds -> say "can't find ABC"
                VA: Front 2 meter (Set target on object, constantly update the instruction util user is in front of the object)
3. Mode 3: Image capturing -> Describe the object in front of the user
                VA: (Say the description)
4. Mode 0: Default mode is Iddle (DEFAULT MODE is 0)
               Iddle mode: do nothing until wake up call

Wake up call is Hey Aura