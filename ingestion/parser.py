from __future__ import annotations
import xml.etree.ElementTree as ET


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root     = ET.fromstring(xml_text)
    articles = root.findall(".//PubmedArticle")
    records  = []

    for art in articles:
        pmid = (art.findtext(".//PMID") or "").strip()

        title_el = art.find(".//ArticleTitle")
        title    = "".join(title_el.itertext()).strip() if title_el is not None else ""

        abstract_parts = []
        for ab in art.findall(".//AbstractText"):
            label = ab.get("Label")
            text  = "".join(ab.itertext()).strip()
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts)

        pub_types = [pt.text or "" for pt in art.findall(".//PublicationType")]
        if any("Retracted" in pt for pt in pub_types):
            continue

        authors = []
        for author in art.findall(".//Author"):
            last = author.findtext("LastName") or ""
            init = author.findtext("Initials") or ""
            if last:
                authors.append(f"{last} {init}".strip())

        journal = (
            art.findtext(".//Journal/Title")
            or art.findtext(".//ISOAbbreviation")
            or ""
        )

        year = None
        year_el = art.find(".//PubDate/Year")
        if year_el is not None:
            try: year = int(year_el.text)
            except: pass
        if year is None:
            med_el = art.find(".//PubDate/MedlineDate")
            if med_el is not None:
                try: year = int((med_el.text or "")[:4])
                except: pass

        mesh_terms = [
            mh.findtext("DescriptorName") or ""
            for mh in art.findall(".//MeshHeading")
        ]
        mesh_terms = [m for m in mesh_terms if m]

        doi = None
        for aid in art.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text

        records.append({
            "pmid":       pmid,
            "title":      title,
            "abstract":   abstract,
            "authors":    authors,
            "journal":    journal,
            "year":       year,
            "mesh_terms": mesh_terms,
            "doi":        doi,
        })

    return records