"""Tests for ``app/utils/sanitization.py``."""

import pytest

from app.utils.sanitization import (
    sanitize_dict,
    sanitize_email,
    sanitize_list,
    sanitize_string,
    validate_password_strength,
)


class TestSanitizeString:
    def test_plain_text_is_unchanged(self):
        assert sanitize_string("hello world 123") == "hello world 123"

    def test_html_special_characters_are_escaped(self):
        assert sanitize_string("<b>Tom & \"Jerry\" 'x'</b>") == (
            "&lt;b&gt;Tom &amp; &quot;Jerry&quot; &#x27;x&#x27;&lt;/b&gt;"
        )

    def test_script_block_and_its_contents_are_removed(self):
        assert sanitize_string("before<script>alert(1)</script>after") == "beforeafter"

    def test_script_block_with_attributes_and_newlines_is_removed(self):
        value = 'a<script type="text/javascript">\nalert(1);\n</script>b'

        assert sanitize_string(value) == "ab"

    def test_multiple_script_blocks_are_all_removed(self):
        assert sanitize_string("<script>a</script>mid<script>b</script>") == "mid"

    def test_null_bytes_are_removed(self):
        assert sanitize_string("a\0b\0c") == "abc"

    @pytest.mark.parametrize(
        "value",
        [
            "<img src=x onerror=alert(1)>",
            "<SCRIPT>alert(1)</SCRIPT>",
            "<svg/onload=alert(1)>",
            "\"><script>alert(1)</script>",
            "<a href='javascript:alert(1)'>x</a>",
            "<scr\0ipt>alert(1)</scr\0ipt>",
        ],
    )
    def test_output_never_contains_raw_angle_brackets(self, value):
        result = sanitize_string(value)

        assert "<" not in result
        assert ">" not in result

    @pytest.mark.parametrize("value, expected", [(42, "42"), (None, "None"), (3.5, "3.5"), (True, "True")])
    def test_non_strings_are_converted(self, value, expected):
        assert sanitize_string(value) == expected

    def test_empty_string(self):
        assert sanitize_string("") == ""


class TestSanitizeEmail:
    @pytest.mark.parametrize(
        "email, expected",
        [
            ("user@example.com", "user@example.com"),
            ("User.Name+tag@Example.COM", "user.name+tag@example.com"),
            ("a_b-c%d@sub.domain.co.uk", "a_b-c%d@sub.domain.co.uk"),
        ],
    )
    def test_valid_addresses_are_normalised_to_lowercase(self, email, expected):
        assert sanitize_email(email) == expected

    @pytest.mark.parametrize(
        "email",
        [
            "",
            "plainaddress",
            "@example.com",
            "user@",
            "user@example",
            "user@example.c",
            "user name@example.com",
            "user@exa mple.com",
            "user@@example.com",
            "<script>@example.com",
            "user@example.com\n",  # regression: "$" also matched before a trailing newline
            "user@example.com\r\n",
        ],
    )
    def test_invalid_addresses_are_rejected(self, email):
        with pytest.raises(ValueError, match="Invalid email format"):
            sanitize_email(email)

    def test_embedded_script_block_is_stripped_before_validation(self):
        assert sanitize_email("user@example.com<script>alert(1)</script>") == "user@example.com"


class TestSanitizeDict:
    def test_sanitizes_string_values(self):
        assert sanitize_dict({"a": "<b>", "b": "ok"}) == {"a": "&lt;b&gt;", "b": "ok"}

    def test_recurses_into_nested_dicts_and_lists(self):
        data = {"outer": {"inner": "<i>", "items": ["<a>", {"deep": "<d>"}, ["<n>"]]}}

        assert sanitize_dict(data) == {
            "outer": {"inner": "&lt;i&gt;", "items": ["&lt;a&gt;", {"deep": "&lt;d&gt;"}, ["&lt;n&gt;"]]}
        }

    def test_non_string_values_pass_through_untouched(self):
        data = {"n": 1, "f": 2.5, "b": True, "none": None}

        assert sanitize_dict(data) == data

    def test_does_not_mutate_the_input(self):
        data = {"a": "<b>", "nested": {"c": "<c>"}, "list": ["<l>"]}

        sanitize_dict(data)

        assert data == {"a": "<b>", "nested": {"c": "<c>"}, "list": ["<l>"]}

    def test_empty_dict(self):
        assert sanitize_dict({}) == {}


class TestSanitizeList:
    def test_sanitizes_strings_and_recurses(self):
        assert sanitize_list(["<a>", {"k": "<b>"}, ["<c>"], 7, None]) == [
            "&lt;a&gt;",
            {"k": "&lt;b&gt;"},
            ["&lt;c&gt;"],
            7,
            None,
        ]

    def test_does_not_mutate_the_input(self):
        data = ["<a>", ["<b>"]]

        sanitize_list(data)

        assert data == ["<a>", ["<b>"]]

    def test_empty_list(self):
        assert sanitize_list([]) == []


class TestValidatePasswordStrength:
    def test_strong_password_is_accepted(self):
        assert validate_password_strength("Str0ng!Pass") is True

    def test_minimum_length_boundary(self):
        assert validate_password_strength("Abcde1!x") is True  # exactly 8
        with pytest.raises(ValueError, match="at least 8 characters"):
            validate_password_strength("Abcd1!x")  # 7

    @pytest.mark.parametrize(
        "password, message",
        [
            ("short1!", "at least 8 characters"),
            ("lowercase1!", "uppercase"),
            ("UPPERCASE1!", "lowercase"),
            ("NoNumbers!!", "number"),
            ("NoSpecial123", "special character"),
        ],
    )
    def test_each_rule_is_enforced_with_a_specific_message(self, password, message):
        with pytest.raises(ValueError, match=message):
            validate_password_strength(password)

    @pytest.mark.parametrize("special", list('!@#$%^&*(),.?":{}|<>'))
    def test_every_documented_special_character_counts(self, special):
        assert validate_password_strength(f"Passw0rd{special}") is True

    def test_empty_password_fails_on_length_first(self):
        with pytest.raises(ValueError, match="at least 8 characters"):
            validate_password_strength("")
