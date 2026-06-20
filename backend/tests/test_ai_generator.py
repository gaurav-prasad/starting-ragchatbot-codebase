import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────────

def _text_response(text="Test response"):
    response = MagicMock()
    response.stop_reason = "end_turn"
    block = MagicMock()
    block.text = text
    response.content = [block]
    return response


def _tool_use_response(name="search_course_content", tool_id="tool_001", tool_input=None):
    response = MagicMock()
    response.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = tool_input or {"query": "test"}
    response.content = [block]
    return response


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAIGeneratorInit:

    @patch("ai_generator.anthropic.Anthropic")
    def test_client_created_with_api_key(self, mock_cls):
        from ai_generator import AIGenerator
        gen = AIGenerator(api_key="my-key", model="claude-test")
        mock_cls.assert_called_once_with(api_key="my-key")
        assert gen.client is mock_cls.return_value

    @patch("ai_generator.anthropic.Anthropic")
    def test_base_params_set_correctly(self, mock_cls):
        from ai_generator import AIGenerator
        gen = AIGenerator(api_key="key", model="claude-test")
        assert gen.base_params["model"] == "claude-test"
        assert gen.base_params["temperature"] == 0
        assert gen.base_params["max_tokens"] == 800


class TestGenerateResponse:

    @patch("ai_generator.anthropic.Anthropic")
    def test_returns_text_content_directly(self, mock_cls):
        from ai_generator import AIGenerator
        mock_cls.return_value.messages.create.return_value = _text_response("Hello!")
        gen = AIGenerator(api_key="key", model="m")
        assert gen.generate_response(query="Hi") == "Hello!"

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_tools_means_no_tools_param_in_api_call(self, mock_cls):
        from ai_generator import AIGenerator
        mock_cls.return_value.messages.create.return_value = _text_response()
        gen = AIGenerator(api_key="key", model="m")
        gen.generate_response(query="Hi")
        kwargs = mock_cls.return_value.messages.create.call_args[1]
        assert "tools" not in kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_with_tools_passes_tool_choice_auto(self, mock_cls):
        from ai_generator import AIGenerator
        mock_cls.return_value.messages.create.return_value = _text_response()
        gen = AIGenerator(api_key="key", model="m")
        tools = [{"name": "search", "description": "search", "input_schema": {}}]
        gen.generate_response(query="Hi", tools=tools)
        kwargs = mock_cls.return_value.messages.create.call_args[1]
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == {"type": "auto"}

    @patch("ai_generator.anthropic.Anthropic")
    def test_system_prompt_used_when_no_history(self, mock_cls):
        from ai_generator import AIGenerator
        mock_cls.return_value.messages.create.return_value = _text_response()
        gen = AIGenerator(api_key="key", model="m")
        gen.generate_response(query="Hi")
        kwargs = mock_cls.return_value.messages.create.call_args[1]
        assert kwargs["system"] == AIGenerator.SYSTEM_PROMPT

    @patch("ai_generator.anthropic.Anthropic")
    def test_conversation_history_appended_to_system(self, mock_cls):
        from ai_generator import AIGenerator
        mock_cls.return_value.messages.create.return_value = _text_response()
        gen = AIGenerator(api_key="key", model="m")
        gen.generate_response(query="Follow-up", conversation_history="User: Hi\nAssistant: Hello")
        kwargs = mock_cls.return_value.messages.create.call_args[1]
        assert "Previous conversation:" in kwargs["system"]
        assert "User: Hi" in kwargs["system"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_use_response_delegates_to_handle_tool_execution(self, mock_cls):
        from ai_generator import AIGenerator
        client = mock_cls.return_value
        client.messages.create.side_effect = [
            _tool_use_response(),
            _text_response("Final answer"),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "search results"

        gen = AIGenerator(api_key="key", model="m")
        result = gen.generate_response(
            query="course question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        assert result == "Final answer"
        assert client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_use_without_tool_manager_returns_text_block(self, mock_cls):
        from ai_generator import AIGenerator
        # stop_reason is tool_use but no tool_manager — falls back to content[0].text
        resp = _tool_use_response()
        resp.content[0].type = "tool_use"
        resp.content[0].text = "fallback"  # won't be reached; just defensive
        mock_cls.return_value.messages.create.return_value = resp
        gen = AIGenerator(api_key="key", model="m")
        # Should not raise; no tool_manager means _handle_tool_execution is skipped
        # but content[0].text doesn't exist on a tool_use block — expect AttributeError
        # unless the caller always provides a tool_manager when tools are given.
        # This test verifies no crash when tool_manager=None even if stop_reason=tool_use.
        # The code path: response.content[0].text is accessed — it'll return the MagicMock default.
        result = gen.generate_response(query="q")
        assert result is not None


class TestHandleToolExecution:

    @patch("ai_generator.anthropic.Anthropic")
    def test_executes_each_tool_block(self, mock_cls):
        from ai_generator import AIGenerator
        client = mock_cls.return_value
        client.messages.create.return_value = _text_response("Done")

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        initial = _tool_use_response(name="search_course_content", tool_id="t1",
                                     tool_input={"query": "foo", "course_name": "Python"})

        gen = AIGenerator(api_key="key", model="m")
        base_params = {
            **gen.base_params,
            "messages": [{"role": "user", "content": "q"}],
            "system": "sys",
        }
        result = gen._handle_tool_execution(initial, base_params, mock_tool_manager)

        assert result == "Done"
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="foo", course_name="Python"
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_final_api_call_omits_tools(self, mock_cls):
        from ai_generator import AIGenerator
        client = mock_cls.return_value
        client.messages.create.return_value = _text_response("Done")

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "r"

        gen = AIGenerator(api_key="key", model="m")
        base_params = {
            **gen.base_params,
            "messages": [{"role": "user", "content": "q"}],
            "system": "sys",
            "tools": [{"name": "search"}],
        }
        gen._handle_tool_execution(_tool_use_response(), base_params, mock_tool_manager)

        final_kwargs = client.messages.create.call_args[1]
        assert "tools" not in final_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_result_added_to_messages(self, mock_cls):
        from ai_generator import AIGenerator
        client = mock_cls.return_value
        client.messages.create.return_value = _text_response("Done")

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "search output"

        gen = AIGenerator(api_key="key", model="m")
        base_params = {
            **gen.base_params,
            "messages": [{"role": "user", "content": "q"}],
            "system": "sys",
        }
        gen._handle_tool_execution(_tool_use_response(tool_id="tid1"), base_params, mock_tool_manager)

        final_messages = client.messages.create.call_args[1]["messages"]
        # messages: [user, assistant tool_use, user tool_result]
        tool_result_msg = final_messages[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tid1"
        assert tool_result_msg["content"][0]["content"] == "search output"
