import replicate

def get_response(llm, smell_prompt, prompt_input, max_length):
    output = replicate.run(llm, input={"prompt": f"{smell_prompt} {prompt_input} Assistant: ",
                                  "temperature":0.9, "top_p":0.7, "max_length":max_length, "repetition_penalty":1})
    return output
