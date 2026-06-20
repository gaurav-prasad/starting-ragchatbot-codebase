import pytest
from unittest.mock import MagicMock, patch


# ── Patch helpers ─────────────────────────────────────────────────────────────
# Decorators applied bottom-up; method arg order matches bottom→top:
# CourseSearchTool, ToolManager, SessionManager, AIGenerator, VectorStore, DocumentProcessor

PATCHES = [
    "rag_system.DocumentProcessor",
    "rag_system.VectorStore",
    "rag_system.AIGenerator",
    "rag_system.SessionManager",
    "rag_system.ToolManager",
    "rag_system.CourseSearchTool",
]


def _apply_patches(fn):
    for p in reversed(PATCHES):
        fn = patch(p)(fn)
    return fn


def _make_config():
    cfg = MagicMock()
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.CHROMA_PATH = "./test_chroma"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.MAX_RESULTS = 5
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-test"
    cfg.MAX_HISTORY = 2
    return cfg


# ── Init ──────────────────────────────────────────────────────────────────────

class TestRAGSystemInit:

    @_apply_patches
    def test_all_components_instantiated(self, mock_search_cls, mock_tm_cls,
                                          mock_sm_cls, mock_ai_cls, mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        cfg = _make_config()
        RAGSystem(cfg)
        mock_dp_cls.assert_called_once_with(cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
        mock_vs_cls.assert_called_once_with(cfg.CHROMA_PATH, cfg.EMBEDDING_MODEL, cfg.MAX_RESULTS)
        mock_ai_cls.assert_called_once_with(cfg.ANTHROPIC_API_KEY, cfg.ANTHROPIC_MODEL)
        mock_sm_cls.assert_called_once_with(cfg.MAX_HISTORY)

    @_apply_patches
    def test_search_tool_registered_with_tool_manager(self, mock_search_cls, mock_tm_cls,
                                                        mock_sm_cls, mock_ai_cls,
                                                        mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        mock_tm_cls.return_value.register_tool.assert_called_once_with(
            mock_search_cls.return_value
        )


# ── add_course_document ───────────────────────────────────────────────────────

class TestAddCourseDocument:

    @_apply_patches
    def test_success_returns_course_and_chunk_count(self, mock_search_cls, mock_tm_cls,
                                                     mock_sm_cls, mock_ai_cls,
                                                     mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        from models import Course, CourseChunk
        rag = RAGSystem(_make_config())
        course = Course(title="Python 101")
        chunks = [CourseChunk(content="c", course_title="Python 101", chunk_index=i) for i in range(3)]
        rag.document_processor.process_course_document.return_value = (course, chunks)

        result_course, count = rag.add_course_document("/path/course.txt")

        assert result_course == course
        assert count == 3
        rag.vector_store.add_course_metadata.assert_called_once_with(course)
        rag.vector_store.add_course_content.assert_called_once_with(chunks)

    @_apply_patches
    def test_exception_returns_none_and_zero(self, mock_search_cls, mock_tm_cls,
                                              mock_sm_cls, mock_ai_cls,
                                              mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.document_processor.process_course_document.side_effect = Exception("parse error")

        result_course, count = rag.add_course_document("/bad/path.txt")

        assert result_course is None
        assert count == 0


# ── add_course_folder ─────────────────────────────────────────────────────────

class TestAddCourseFolder:

    @_apply_patches
    def test_nonexistent_folder_returns_zeros(self, mock_search_cls, mock_tm_cls,
                                               mock_sm_cls, mock_ai_cls,
                                               mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        courses, chunks = rag.add_course_folder("/does/not/exist")
        assert courses == 0
        assert chunks == 0

    @_apply_patches
    def test_clear_existing_calls_clear_all_data(self, mock_search_cls, mock_tm_cls,
                                                  mock_sm_cls, mock_ai_cls,
                                                  mock_vs_cls, mock_dp_cls, tmp_path):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.vector_store.get_existing_course_titles.return_value = []
        rag.add_course_folder(str(tmp_path), clear_existing=True)
        rag.vector_store.clear_all_data.assert_called_once()

    @_apply_patches
    def test_clear_false_does_not_clear_data(self, mock_search_cls, mock_tm_cls,
                                              mock_sm_cls, mock_ai_cls,
                                              mock_vs_cls, mock_dp_cls, tmp_path):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.vector_store.get_existing_course_titles.return_value = []
        rag.add_course_folder(str(tmp_path), clear_existing=False)
        rag.vector_store.clear_all_data.assert_not_called()

    @_apply_patches
    def test_existing_course_is_skipped(self, mock_search_cls, mock_tm_cls,
                                         mock_sm_cls, mock_ai_cls,
                                         mock_vs_cls, mock_dp_cls, tmp_path):
        from rag_system import RAGSystem
        from models import Course
        (tmp_path / "course.txt").write_text("dummy")
        rag = RAGSystem(_make_config())
        rag.document_processor.process_course_document.return_value = (
            Course(title="Python 101"), []
        )
        rag.vector_store.get_existing_course_titles.return_value = ["Python 101"]

        courses, _ = rag.add_course_folder(str(tmp_path))

        assert courses == 0
        rag.vector_store.add_course_metadata.assert_not_called()

    @_apply_patches
    def test_new_course_added_to_vector_store(self, mock_search_cls, mock_tm_cls,
                                               mock_sm_cls, mock_ai_cls,
                                               mock_vs_cls, mock_dp_cls, tmp_path):
        from rag_system import RAGSystem
        from models import Course, CourseChunk
        (tmp_path / "course.txt").write_text("dummy")
        rag = RAGSystem(_make_config())
        course = Course(title="New Course")
        chunks = [CourseChunk(content="c", course_title="New Course", chunk_index=0)]
        rag.document_processor.process_course_document.return_value = (course, chunks)
        rag.vector_store.get_existing_course_titles.return_value = []

        courses, chunk_count = rag.add_course_folder(str(tmp_path))

        assert courses == 1
        assert chunk_count == 1
        rag.vector_store.add_course_metadata.assert_called_once_with(course)
        rag.vector_store.add_course_content.assert_called_once_with(chunks)

    @_apply_patches
    def test_non_txt_files_are_ignored(self, mock_search_cls, mock_tm_cls,
                                        mock_sm_cls, mock_ai_cls,
                                        mock_vs_cls, mock_dp_cls, tmp_path):
        from rag_system import RAGSystem
        (tmp_path / "image.png").write_text("not a doc")
        rag = RAGSystem(_make_config())
        rag.vector_store.get_existing_course_titles.return_value = []

        courses, _ = rag.add_course_folder(str(tmp_path))

        assert courses == 0
        rag.document_processor.process_course_document.assert_not_called()


# ── query ─────────────────────────────────────────────────────────────────────

class TestQuery:

    @_apply_patches
    def test_returns_response_and_sources(self, mock_search_cls, mock_tm_cls,
                                           mock_sm_cls, mock_ai_cls,
                                           mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.ai_generator.generate_response.return_value = "Great answer"
        rag.tool_manager.get_last_sources.return_value = ["Course A"]

        response, sources = rag.query("What is Python?")

        assert response == "Great answer"
        assert sources == ["Course A"]

    @_apply_patches
    def test_no_session_id_skips_history_and_exchange(self, mock_search_cls, mock_tm_cls,
                                                        mock_sm_cls, mock_ai_cls,
                                                        mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.ai_generator.generate_response.return_value = "Answer"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Question")

        rag.session_manager.get_conversation_history.assert_not_called()
        rag.session_manager.add_exchange.assert_not_called()

    @_apply_patches
    def test_session_id_retrieves_history(self, mock_search_cls, mock_tm_cls,
                                           mock_sm_cls, mock_ai_cls,
                                           mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.session_manager.get_conversation_history.return_value = "User: Hi\nAssistant: Hello"
        rag.ai_generator.generate_response.return_value = "Response"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Follow-up", session_id="session_1")

        rag.session_manager.get_conversation_history.assert_called_once_with("session_1")
        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["conversation_history"] == "User: Hi\nAssistant: Hello"

    @_apply_patches
    def test_session_exchange_saved_after_response(self, mock_search_cls, mock_tm_cls,
                                                    mock_sm_cls, mock_ai_cls,
                                                    mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.ai_generator.generate_response.return_value = "My answer"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("My question", session_id="session_1")

        rag.session_manager.add_exchange.assert_called_once_with(
            "session_1", "My question", "My answer"
        )

    @_apply_patches
    def test_sources_reset_after_retrieval(self, mock_search_cls, mock_tm_cls,
                                            mock_sm_cls, mock_ai_cls,
                                            mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.ai_generator.generate_response.return_value = "r"
        rag.tool_manager.get_last_sources.return_value = ["S"]

        rag.query("q")

        rag.tool_manager.reset_sources.assert_called_once()

    @_apply_patches
    def test_prompt_wraps_original_query(self, mock_search_cls, mock_tm_cls,
                                          mock_sm_cls, mock_ai_cls,
                                          mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.ai_generator.generate_response.return_value = "a"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("What is recursion?")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "What is recursion?" in kwargs["query"]

    @_apply_patches
    def test_tool_definitions_passed_to_ai_generator(self, mock_search_cls, mock_tm_cls,
                                                       mock_sm_cls, mock_ai_cls,
                                                       mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.tool_manager.get_tool_definitions.return_value = [{"name": "search"}]
        rag.ai_generator.generate_response.return_value = "a"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("q")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["tools"] == [{"name": "search"}]


# ── get_course_analytics ──────────────────────────────────────────────────────

class TestGetCourseAnalytics:

    @_apply_patches
    def test_returns_count_and_titles(self, mock_search_cls, mock_tm_cls,
                                       mock_sm_cls, mock_ai_cls,
                                       mock_vs_cls, mock_dp_cls):
        from rag_system import RAGSystem
        rag = RAGSystem(_make_config())
        rag.vector_store.get_course_count.return_value = 2
        rag.vector_store.get_existing_course_titles.return_value = ["A", "B"]

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 2
        assert analytics["course_titles"] == ["A", "B"]
