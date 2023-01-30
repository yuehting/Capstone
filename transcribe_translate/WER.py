import pandas as pd
import sklearn.metrics

split_audio_transcription = pd.read_csv('./output/split_audio_transcription.csv')
split_audio_transcription_en = split_audio_transcription.loc[split_audio_transcription.language == "en"]
split_audio_transcription_en = split_audio_transcription_en[['split_wav_file_name', 'transcription']]
print(split_audio_transcription_en)

gold = pd.read_csv('./materials for report/split_audio_transcription_translation_ENGLISH_GOLD.csv')
gold = gold[['split_wav_file_name', 'transcription']]
gold = gold.rename(columns={'transcription': 'gold_transcription'})
print(gold)
unique_transcription = gold.loc[gold.gold_transcription != "REPEATED"]
unique_transcription = unique_transcription.loc[unique_transcription.gold_transcription != "REPEATED FROM THE START"]
print(unique_transcription)

dataframe = pd.merge(unique_transcription, split_audio_transcription_en, left_index=True, right_index=True)
print(dataframe)
dataframe = dataframe[['split_wav_file_name_x','transcription', 'gold_transcription']]
dataframe = dataframe.rename(columns={'split_wav_file_name_x': 'split_wav_file_name'})
dataframe = dataframe.dropna()
dataframe = dataframe.reset_index(drop=True)
print(dataframe)

def wer(ref, hyp, debug=True):
    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    # return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round((numSub + numDel + numIns) / (float)(len(r)), 3)
    return {'WER': wer_result, 'numCor': numCor, 'numSub': numSub, 'numIns': numIns, 'numDel': numDel,
            "numCount": len(r)}
print(wer(dataframe.gold_transcription[1], dataframe.transcription[1]))
print(dataframe)
dataframe["WER"] = None

for i in range(len(dataframe)):
    result = wer(dataframe.gold_transcription[i], dataframe.transcription[i])
    dataframe["WER"][i] = result['WER']
print(dataframe)
dataframe.to_csv('./output/WER_english_transcription.csv', index=False)
