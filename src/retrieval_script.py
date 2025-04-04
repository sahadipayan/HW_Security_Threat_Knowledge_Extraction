from rag_agent import RAGAgent

def get_retrived_data(topic):

    try:
        rag_agent = RAGAgent(topic)
        user_query = f"""
        What are the possible attack secanrios in {topic}?
        """
        response = rag_agent.handle_query(user_query)
        print(f"Relevant information collected for topic '{topic}'.")
    except Exception as e:
        print(f"failed: Error occurred while collecting information for topic '{topic}' - {e}")

    return response
