from collections import defaultdict

class ContextManager:
    def __init__(self):
        self.sessions = defaultdict(list)  # session_id -> messages
        self.profile = {}                 # user_id -> profile dict

    def add_message(self, session_id: str, role: str, content: str):
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str, limit: int = 20):
        return self.sessions[session_id][-limit:]

    def set_profile(self, user_id: str, **kwargs):
        self.profile[user_id] = {**self.profile.get(user_id, {}), **kwargs}

    def get_profile(self, user_id: str):
        return self.profile.get(user_id, {})
