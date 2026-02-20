# Constants

VOCABLE_FILLERS=["hmm","mm","mhm", "mmm", "uh", "um", "erm", "er"] # stripped from whisper training data, from https://cdn.openai.com/papers/whisper.pdf
LEXICAL_FILLERS=["so", "like", "you know", "sort of", "kinda", "kind of", "basically"]

# Calculate proportion of filler words to non-filler words in a text string
#
# Input: Text string which may contain filler words
# Output: Proportion of filler words in range [0,1]
def calculateFillerProportion(text):
    fillerCount = 0

    for filler in VOCABLE_FILLERS:
        fillerCount += text.count(filler)

    # TODO: Only contextually a filler
    # Seperate as it depends on context if they are fillers, may revisit
    #for filler in LEXICAL_FILLERS:
    #    count += text.count(filler)

    tokens = text.split()
    wordCount = len(tokens)

    return fillerCount / wordCount
