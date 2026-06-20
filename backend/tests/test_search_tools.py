import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_results(docs, metadata=None):
    meta = metadata or [{}] * len(docs)
    return SearchResults(documents=docs, metadata=meta, distances=[0.1] * len(docs))


def _make_store(docs=None, metadata=None, error=None):
    store = MagicMock()
    if error:
        store.search.return_value = SearchResults.empty(error)
    elif docs is not None:
        store.search.return_value = _make_results(docs, metadata)
    else:
        store.search.return_value = _make_results([])
    return store


# ── CourseSearchTool ──────────────────────────────────────────────────────────

class TestCourseSearchToolDefinition:

    def test_name_is_search_course_content(self):
        tool = CourseSearchTool(MagicMock())
        assert tool.get_tool_definition()["name"] == "search_course_content"

    def test_query_is_required(self):
        tool = CourseSearchTool(MagicMock())
        schema = tool.get_tool_definition()["input_schema"]
        assert "query" in schema["required"]

    def test_optional_fields_present(self):
        tool = CourseSearchTool(MagicMock())
        props = tool.get_tool_definition()["input_schema"]["properties"]
        assert "course_name" in props
        assert "lesson_number" in props


class TestCourseSearchToolExecute:

    def test_calls_store_search_with_correct_args(self):
        store = _make_store(docs=["content"], metadata=[{"course_title": "C", "lesson_number": 1}])
        tool = CourseSearchTool(store)
        tool.execute(query="topic", course_name="C", lesson_number=1)
        store.search.assert_called_once_with(query="topic", course_name="C", lesson_number=1)

    def test_no_filters_passes_none_values(self):
        store = _make_store(docs=["content"], metadata=[{"course_title": "C"}])
        tool = CourseSearchTool(store)
        tool.execute(query="topic")
        store.search.assert_called_once_with(query="topic", course_name=None, lesson_number=None)

    def test_returns_formatted_content_on_results(self):
        store = _make_store(
            docs=["Python is great"],
            metadata=[{"course_title": "Python 101", "lesson_number": 2}],
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="python")
        assert "Python 101" in result
        assert "Python is great" in result

    def test_empty_results_no_filters(self):
        tool = CourseSearchTool(_make_store(docs=[]))
        result = tool.execute(query="nothing")
        assert result == "No relevant content found."

    def test_empty_results_with_course_filter(self):
        tool = CourseSearchTool(_make_store(docs=[]))
        result = tool.execute(query="nothing", course_name="ML Course")
        assert "No relevant content found" in result
        assert "ML Course" in result

    def test_empty_results_with_lesson_filter(self):
        tool = CourseSearchTool(_make_store(docs=[]))
        result = tool.execute(query="nothing", lesson_number=3)
        assert "No relevant content found" in result
        assert "lesson 3" in result

    def test_returns_error_string_from_store(self):
        tool = CourseSearchTool(_make_store(error="DB connection failed"))
        result = tool.execute(query="test")
        assert result == "DB connection failed"

    def test_stores_sources_after_search(self):
        store = _make_store(
            docs=["A", "B"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None},
            ],
        )
        tool = CourseSearchTool(store)
        tool.execute(query="test")
        assert "Course A - Lesson 1" in tool.last_sources
        assert "Course B" in tool.last_sources

    def test_sources_empty_before_any_search(self):
        tool = CourseSearchTool(MagicMock())
        assert tool.last_sources == []


class TestCourseSearchToolFormatResults:

    def test_includes_course_and_lesson_in_header(self):
        results = _make_results(
            ["Lesson content"],
            [{"course_title": "ML Basics", "lesson_number": 4}],
        )
        tool = CourseSearchTool(MagicMock())
        output = tool._format_results(results)
        assert "[ML Basics - Lesson 4]" in output
        assert "Lesson content" in output

    def test_omits_lesson_from_header_when_none(self):
        results = _make_results(
            ["Intro text"],
            [{"course_title": "Intro Course"}],
        )
        tool = CourseSearchTool(MagicMock())
        output = tool._format_results(results)
        assert "[Intro Course]" in output
        assert "Lesson" not in output

    def test_multiple_results_joined_by_blank_line(self):
        results = _make_results(
            ["Doc A", "Doc B"],
            [{"course_title": "C", "lesson_number": 1}, {"course_title": "C", "lesson_number": 2}],
        )
        tool = CourseSearchTool(MagicMock())
        output = tool._format_results(results)
        assert "\n\n" in output


# ── ToolManager ───────────────────────────────────────────────────────────────

def _mock_tool(name="test_tool", sources=None):
    tool = MagicMock()
    tool.get_tool_definition.return_value = {
        "name": name,
        "description": "desc",
        "input_schema": {"type": "object"},
    }
    tool.execute.return_value = "tool output"
    tool.last_sources = sources if sources is not None else []
    return tool


class TestToolManager:

    def test_register_tool_stores_by_name(self):
        mgr = ToolManager()
        mgr.register_tool(_mock_tool("my_tool"))
        assert "my_tool" in mgr.tools

    def test_register_tool_raises_when_name_missing(self):
        mgr = ToolManager()
        bad = MagicMock()
        bad.get_tool_definition.return_value = {}
        with pytest.raises(ValueError):
            mgr.register_tool(bad)

    def test_get_tool_definitions_returns_list(self):
        mgr = ToolManager()
        mgr.register_tool(_mock_tool("t1"))
        mgr.register_tool(_mock_tool("t2"))
        defs = mgr.get_tool_definitions()
        assert len(defs) == 2
        names = {d["name"] for d in defs}
        assert names == {"t1", "t2"}

    def test_execute_tool_calls_correct_tool(self):
        mgr = ToolManager()
        tool = _mock_tool("search_course_content")
        mgr.register_tool(tool)
        result = mgr.execute_tool("search_course_content", query="python")
        tool.execute.assert_called_once_with(query="python")
        assert result == "tool output"

    def test_execute_tool_returns_not_found_for_unknown(self):
        mgr = ToolManager()
        result = mgr.execute_tool("ghost_tool")
        assert "not found" in result.lower()

    def test_get_last_sources_returns_non_empty_tool_sources(self):
        mgr = ToolManager()
        mgr.register_tool(_mock_tool("search", sources=["Course A - Lesson 1", "Course B"]))
        assert mgr.get_last_sources() == ["Course A - Lesson 1", "Course B"]

    def test_get_last_sources_returns_empty_when_no_sources(self):
        mgr = ToolManager()
        mgr.register_tool(_mock_tool("search", sources=[]))
        assert mgr.get_last_sources() == []

    def test_reset_sources_clears_all_tools(self):
        mgr = ToolManager()
        tool = _mock_tool("search", sources=["Source 1"])
        mgr.register_tool(tool)
        mgr.reset_sources()
        assert tool.last_sources == []

    def test_reset_sources_no_error_when_no_tools(self):
        mgr = ToolManager()
        mgr.reset_sources()  # should not raise
