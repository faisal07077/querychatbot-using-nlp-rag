# this file is for smoke testing 
# dont use in the inside other folders
#use always in the root structure
#for better answers add multiple questions in this file 
#try to add more questions for the testing


from rag.rag_engine import RAGEngine

rag = RAGEngine()

print(rag.ask("What is the eligibility for B.Tech admission?"))
print(rag.ask("How can I apply for admission?"))
print(rag.ask("What are the college timings?"))
