# Error Analysis

## Case 1
Text: "started off more settled i stayed with it anyway that part surprised me"
Predicted: neutral
Issue: Mixed emotional signals
Reason: Text contains both calm and uncertainty, model averages to neutral
Fix: Use contextual models like BERT

## Case 2
Text: "by teh end i was pretty grounded cafe ambience weirdly helped"
Predicted: neutral
Issue: Misspelled word ("teh")
Reason: TF-IDF cannot handle spelling mistakes well
Fix: Add spell correction or preprocessing

## Case 3
Text: "ok session"
Predicted: restless
Issue: Very short text
Reason: Not enough information for model
Fix: Use metadata more strongly or fallback rules

## Case 4
Text: "by the end i was locked in for a bit i kept thinking about emails"
Predicted: focused
Issue: Conflicting signals (focus + distraction)
Reason: Model confused between focus and stress
Fix: Better feature weighting

## Case 5
Text: "mind racing"
Predicted: neutral
Issue: Ambiguous phrase
Reason: Could mean anxiety but lacks explicit keywords
Fix: Use contextual embeddings

## Case 6
Text: "lowkey felt pretty grounded i had to restart once"
Predicted: calm
Issue: Mixed signals
Reason: Calm + interruption creates confusion
Fix: Sequence-aware models

## Case 7
Text: "ok session"
Predicted: restless
Issue: Repeated short input failure
Reason: Sparse TF-IDF features
Fix: Handle short text separately

## Case 8
Text: "ok session"
Predicted: mixed
Issue: Inconsistent predictions
Reason: Model instability on weak inputs
Fix: Add confidence threshold logic

## Case 9
Text: "lowkey felt kind of jumpy but then my mind wandered again"
Predicted: restless
Issue: Multiple emotional transitions
Reason: Model cannot track sequence
Fix: Use sequential models (LSTM/BERT)

## Case 10
Text: "i noticed i was fine i guess cafe ambience weirdly helped"
Predicted: mixed
Issue: Uncertain tone ("i guess")
Reason: Subtle language not captured by TF-IDF
Fix: Context-aware models