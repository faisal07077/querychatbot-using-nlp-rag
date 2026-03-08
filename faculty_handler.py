# faculty_handler.py
import json


class FacultyHandler:
    """
    Handles all faculty-related queries for CSE department.
    """

    def __init__(self, path="data/faculty.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.faculty = json.load(f)

        # Build name index for fast lookup
        self.name_index = []
        for f in self.faculty:
            parts = [p.lower() for p in f["name"].split() if len(p) > 3]
            self.name_index.append((parts, f))

    def get_hod(self) -> str:
        for f in self.faculty:
            if "hod" in f["designation"].lower():
                return (
                    f"👩‍🏫 HoD of CSE Department:\n\n"
                    f"Name: {f['name']}\n"
                    f"Designation: {f['designation']}\n"
                    f"Qualification: {f['qualification']}\n"
                    f"Experience: {f['experience']}"
                )
        return "HoD information not available."

    def get_all_cse_faculty(self) -> str:
        ug = [f for f in self.faculty if f["department"] == "CSE" and f["program"] == "UG"]
        pg = [f for f in self.faculty if f["department"] == "CSE" and f["program"] == "PG"]

        response = f"📚 CSE Department has {len(self.faculty)} faculty members.\n\n"

        response += "🎓 UG Teaching Staff:\n"
        for f in ug:
            response += f"• {f['name']} — {f['designation']} ({f['experience']})\n"

        response += f"\n🎓 PG Teaching Staff:\n"
        for f in pg:
            response += f"• {f['name']} — {f['designation']} ({f['experience']})\n"

        return response.strip()

    def get_professors(self) -> str:
        profs = [f for f in self.faculty if
                 "professor" in f["designation"].lower() and
                 "assistant" not in f["designation"].lower() and
                 "associate" not in f["designation"].lower()]
        if not profs:
            return "No professor details found."
        response = "👩‍🏫 Professors in CSE Department:\n\n"
        for f in profs:
            response += f"• {f['name']} — {f['designation']}\n"
            response += f"  Qualification: {f['qualification']} | Experience: {f['experience']}\n\n"
        return response.strip()

    def search_by_name(self, query: str) -> str:
        """
        Only match if query contains Dr./Mrs./Ms. prefix 
        OR matches a full last name (5+ chars) exactly.
        This prevents false matches on general questions.
        """
        query_lower = query.lower()

        # Only search if query looks like a name search
        has_title = any(title in query_lower for title in ["dr.", "dr ", "mrs.", "mrs ", "ms.", "ms ", "mr.", "mr "])
        
        if not has_title:
            # Without a title, only match if a long unique name part is present
            for name_parts, f in self.name_index:
                for part in name_parts:
                    if len(part) >= 6 and part in query_lower:
                        return self._format_faculty(f)
            return None

        # With title — search more broadly
        for name_parts, f in self.name_index:
            if any(part in query_lower for part in name_parts if len(part) > 3):
                return self._format_faculty(f)

        return None

    def _format_faculty(self, f: dict) -> str:
        return (
            f"👩‍🏫 Faculty Details:\n\n"
            f"Name: {f['name']}\n"
            f"Designation: {f['designation']}\n"
            f"Department: {f['department']}\n"
            f"Program: {f['program']}\n"
            f"Qualification: {f['qualification']}\n"
            f"Experience: {f['experience']}"
        )

    def get_count(self) -> str:
        ug_count = len([f for f in self.faculty if f["program"] == "UG"])
        pg_count = len([f for f in self.faculty if f["program"] == "PG"])
        return (
            f"CSE Department has a total of {len(self.faculty)} faculty members:\n"
            f"• UG Teaching Staff: {ug_count}\n"
            f"• PG Teaching Staff: {pg_count}"
        )
