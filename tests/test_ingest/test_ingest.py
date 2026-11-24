import httpx
import pandas as pd
import pytest

from issue_classifier.ingest.ingest import (
    add_to_finalDf,
    filter_pr,
    issues_getter,
    next_page,
)

WIDGET_ISSUES_URL = "https://api.example.test/repos/acme/widget/issues"
GADGET_ISSUES_URL = "https://api.example.test/repos/acme/gadget/issues"
EXPECTED_COLUMNS = ["number", "title", "body", "repo", "label"]


def gh_issue(
    number: int,
    title: str = "title",
    body: str | None = "body",
    *,
    pull_request: bool = False,
) -> dict:
    issue = {
        "number": number,
        "title": title,
        "body": body,
        "state": "closed",
        "id": 10_000 + number,
    }
    if pull_request:
        issue["pull_request"] = {
            "url": f"https://api.example.test/repos/acme/widget/pulls/{number}"
        }
    return issue


# --- filter_pr ---------------------------------------------------------------


def test_filter_pr_keeps_issue_and_drops_pull_request():
    kept: list[dict] = []
    issues = [
        gh_issue(1, title="real issue"),
        gh_issue(2, title="this is a PR", pull_request=True),
    ]

    filter_pr(json_aux=issues, json_url=kept)

    assert [item["number"] for item in kept] == [1]
    assert "pull_request" not in kept[0]


def test_filter_pr_empty_list_is_a_no_op():
    kept: list[dict] = []

    filter_pr(json_aux=[], json_url=kept)

    assert kept == []


# --- next_page ---------------------------------------------------------------


def test_next_page_single_page_without_link_header(mock_client, github_headers):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[gh_issue(11), gh_issue(12)])

    collected: list[dict] = []
    with mock_client(handler) as client:
        next_page(
            url=WIDGET_ISSUES_URL,
            headers=github_headers,
            current_params={"labels": "bug", "state": "closed", "per_page": 100},
            client=client,
            json_url=collected,
        )

    assert len(seen) == 1
    assert [item["number"] for item in collected] == [11, 12]


def test_next_page_follows_link_exactly_twice_without_duplicating_params(
    mock_client, github_headers
):
    page1_params = {"labels": "bug", "state": "closed", "per_page": 100}
    next_url = (
        f"{WIDGET_ISSUES_URL}?labels=bug&state=closed&per_page=100&page=2"
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        params = request.url.params
        if params.get("page") == "2":
            assert params.get_list("labels") == ["bug"]
            assert params.get_list("state") == ["closed"]
            assert params.get_list("per_page") == ["100"]
            return httpx.Response(200, json=[gh_issue(22)])
        return httpx.Response(
            200,
            json=[gh_issue(21)],
            headers={"Link": f'<{next_url}>; rel="next"'},
        )

    collected: list[dict] = []
    with mock_client(handler) as client:
        next_page(
            url=WIDGET_ISSUES_URL,
            headers=github_headers,
            current_params=page1_params,
            client=client,
            json_url=collected,
        )

    assert len(seen) == 2
    assert [item["number"] for item in collected] == [21, 22]


def test_next_page_discards_pull_request_on_second_page(mock_client, github_headers):
    next_url = f"{WIDGET_ISSUES_URL}?page=2"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("page") == "2":
            return httpx.Response(
                200,
                json=[
                    gh_issue(32, title="later issue"),
                    gh_issue(33, title="later PR", pull_request=True),
                ],
            )
        return httpx.Response(
            200,
            json=[gh_issue(31)],
            headers={"Link": f'<{next_url}>; rel="next"'},
        )

    collected: list[dict] = []
    with mock_client(handler) as client:
        next_page(
            url=WIDGET_ISSUES_URL,
            headers=github_headers,
            current_params={"labels": "bug", "state": "closed", "per_page": 100},
            client=client,
            json_url=collected,
        )

    assert [item["number"] for item in collected] == [31, 32]


def test_next_page_raises_on_http_500(mock_client, github_headers):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "upstream exploded"})

    with mock_client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        next_page(
            url=WIDGET_ISSUES_URL,
            headers=github_headers,
            current_params={"labels": "bug", "state": "closed", "per_page": 100},
            client=client,
            json_url=[],
        )


# --- add_to_finalDf ----------------------------------------------------------


def test_add_to_finalDf_sets_expected_columns_and_constant_repo_label():
    rows = [
        gh_issue(1, title="one", body="alpha"),
        gh_issue(2, title="two", body="beta"),
    ]

    df = add_to_finalDf(clean_json=rows, repos_list="widget", params_labels="bug")

    assert list(df.columns) == EXPECTED_COLUMNS
    assert df["repo"].tolist() == ["widget", "widget"]
    assert df["label"].tolist() == ["bug", "bug"]
    assert df["number"].tolist() == [1, 2]
    assert df["title"].tolist() == ["one", "two"]
    assert df["body"].tolist() == ["alpha", "beta"]


def test_add_to_finalDf_keeps_missing_body_as_null():
    rows = [gh_issue(7, title="no body", body=None)]

    df = add_to_finalDf(clean_json=rows, repos_list="widget", params_labels="docs")

    assert len(df) == 1
    assert df.loc[0, "number"] == 7
    assert df.loc[0, "title"] == "no body"
    assert pd.isna(df.loc[0, "body"])


def test_add_to_finalDf_empty_list_returns_empty_frame_with_columns():
    df = add_to_finalDf(clean_json=[], repos_list="widget", params_labels="bug")

    assert df.empty
    assert list(df.columns) == EXPECTED_COLUMNS


# --- issues_getter -----------------------------------------------------------


TWO_REPOS_CONF = {
    "widget": {
        "feature": "feat-label",
        "bug": "bug-label",
    },
    "gadget": {
        "feature": "New Feature",
        "bug": "Bug",
    },
}
TWO_REPOS_URLS = [WIDGET_ISSUES_URL, GADGET_ISSUES_URL]


def _payload_for(path: str, github_label: str) -> list[dict]:
    table = {
        ("/repos/acme/widget/issues", "feat-label"): [
            gh_issue(101, title="widget feature")
        ],
        ("/repos/acme/widget/issues", "bug-label"): [
            gh_issue(102, title="widget bug")
        ],
        ("/repos/acme/gadget/issues", "New Feature"): [
            gh_issue(201, title="gadget feature")
        ],
        ("/repos/acme/gadget/issues", "Bug"): [gh_issue(202, title="gadget bug")],
    }
    return table.get((path, github_label), [])


def test_issues_getter_two_repos_two_classes_uses_github_labels(
    mock_client, github_headers
):
    requested_labels: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        github_label = request.url.params.get("labels")
        requested_labels.append((request.url.path, github_label))
        return httpx.Response(
            200, json=_payload_for(request.url.path, github_label)
        )

    with mock_client(handler) as client:
        df = issues_getter(
            repos_conf=TWO_REPOS_CONF,
            url_list=TWO_REPOS_URLS,
            headers=github_headers,
            client=client,
        )

    assert set(requested_labels) == {
        ("/repos/acme/widget/issues", "feat-label"),
        ("/repos/acme/widget/issues", "bug-label"),
        ("/repos/acme/gadget/issues", "New Feature"),
        ("/repos/acme/gadget/issues", "Bug"),
    }
    assert len(df) == 4
    got = set(zip(df["repo"], df["label"], df["number"], df["title"]))
    assert got == {
        ("widget", "feature", 101, "widget feature"),
        ("widget", "bug", 102, "widget bug"),
        ("gadget", "feature", 201, "gadget feature"),
        ("gadget", "bug", 202, "gadget bug"),
    }


def test_issues_getter_drops_same_number_in_two_classes_of_same_repo(
    mock_client, github_headers
):
    def handler(request: httpx.Request) -> httpx.Response:
        label = request.url.params.get("labels")
        if label == "feat-label":
            return httpx.Response(
                200,
                json=[
                    gh_issue(42, title="shared as feature"),
                    gh_issue(1, title="only feature"),
                ],
            )
        if label == "bug-label":
            return httpx.Response(
                200,
                json=[
                    gh_issue(42, title="shared as bug"),
                    gh_issue(2, title="only bug"),
                ],
            )
        return httpx.Response(200, json=[])

    conf = {"widget": {"feature": "feat-label", "bug": "bug-label"}}
    with mock_client(handler) as client:
        df = issues_getter(
            repos_conf=conf,
            url_list=[WIDGET_ISSUES_URL],
            headers=github_headers,
            client=client,
        )

    assert set(df["number"]) == {1, 2}
    assert 42 not in set(df["number"])


def test_issues_getter_keeps_same_number_in_different_repos(
    mock_client, github_headers
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/widget/issues"):
            return httpx.Response(200, json=[gh_issue(7, title="widget seven")])
        if request.url.path.endswith("/gadget/issues"):
            return httpx.Response(200, json=[gh_issue(7, title="gadget seven")])
        return httpx.Response(200, json=[])

    conf = {
        "widget": {"bug": "bug-label"},
        "gadget": {"bug": "Bug"},
    }
    with mock_client(handler) as client:
        df = issues_getter(
            repos_conf=conf,
            url_list=TWO_REPOS_URLS,
            headers=github_headers,
            client=client,
        )

    assert len(df) == 2
    assert set(zip(df["repo"], df["number"], df["title"])) == {
        ("widget", 7, "widget seven"),
        ("gadget", 7, "gadget seven"),
    }


def test_issues_getter_skips_class_with_empty_issue_list(mock_client, github_headers):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("labels") == "feat-label":
            return httpx.Response(200, json=[gh_issue(9, title="only feature")])
        return httpx.Response(200, json=[])

    conf = {"widget": {"feature": "feat-label", "bug": "bug-label"}}
    with mock_client(handler) as client:
        df = issues_getter(
            repos_conf=conf,
            url_list=[WIDGET_ISSUES_URL],
            headers=github_headers,
            client=client,
        )

    assert len(df) == 1
    assert df.loc[0, "number"] == 9
    assert df.loc[0, "label"] == "feature"
    assert df.loc[0, "repo"] == "widget"


def test_issues_getter_all_empty_returns_empty_dataframe_with_columns(
    mock_client, github_headers
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with mock_client(handler) as client:
        df = issues_getter(
            repos_conf=TWO_REPOS_CONF,
            url_list=TWO_REPOS_URLS,
            headers=github_headers,
            client=client,
        )

    assert df.empty
    assert list(df.columns) == EXPECTED_COLUMNS
