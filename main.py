from agent import Agent

def write_file(name):
    with open(name.name+"_mem.txt","w") as f:
        for mems in name.memory_stream.memory_objects:
            f.write(mems.text+"\n")

if __name__=="__main__":
    Holmes = Agent(name = "Sherlock Holmes", context = """
    You are Sherlock Holmes. Your task is to reply to whatever Watson says just like Sherlock Holmes would. If Watson hasn't said anything to you, you say "Good day, Watson!"
    """)
    Watson = Agent(name = "John H. Watson", context = """
    You are John H. Watson. Your task is to reply to Holmes just like Watson would. If Holmes hasn't said anything to you, you say "Good day, Holmes! Have you read the newspaper
    lately?"
    """
    ,initiate_conversation = True)

    flag = 0
    h,w = None, None

    if Watson.initiate_conversation:
        while flag<5:
            w = Watson.chat(h)
            h = Holmes.chat(w)
            flag+=1
            if flag == 5:
                break


    if Holmes.initiate_conversation:
        while flag<5:
            h = Holmes.chat(w)
            w = Watson.chat(h)
            flag+=1
            if flag == 5:
                break

    write_file(name=Holmes)
    write_file(name=Watson)
    