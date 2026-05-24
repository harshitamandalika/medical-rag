from ingestion.parser import parse_pubmed_xml

MINIMAL_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Test Article</ArticleTitle>
        <Abstract>
          <AbstractText>This is a test abstract.</AbstractText>
        </Abstract>
        <Journal>
          <Title>Test Journal</Title>
          <JournalIssue>
            <PubDate><Year>2024</Year></PubDate>
          </JournalIssue>
        </Journal>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <Initials>J</Initials>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_parse_returns_records():
    records = parse_pubmed_xml(MINIMAL_XML)
    assert len(records) == 1


def test_parse_fields():
    records = parse_pubmed_xml(MINIMAL_XML)
    r = records[0]
    assert r["pmid"] == "12345678"
    assert r["title"] == "Test Article"
    assert "test abstract" in r["abstract"].lower()
    assert r["year"] == 2024
    assert r["authors"] == ["Smith J"]


def test_parse_empty_xml():
    records = parse_pubmed_xml("<PubmedArticleSet></PubmedArticleSet>")
    assert records == []