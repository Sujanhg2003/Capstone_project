from collections import defaultdict, deque


class ConversationMemory:
    def __init__(self, max_turns=5):
        self.sessions = defaultdict(
            lambda: deque(maxlen=max_turns)
        )

    def get_history(self, session_id):
        return list(self.sessions[session_id])

    def add_turn(self, session_id, question, answer):
        self.sessions[session_id].append({
            "question": question,
            "answer": answer,
        })

    def clear(self, session_id):
        self.sessions.pop(session_id, None)