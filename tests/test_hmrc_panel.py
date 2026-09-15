from analysis.download_hmrc_panel import iter_months, query_url


def test_iter_months_crosses_year_boundary() -> None:
    assert list(iter_months(202311, 202402)) == [202311, 202312, 202401, 202402]


def test_hmrc_query_preserves_product_destination_grain() -> None:
    url = query_url(202504, 400)
    assert "MonthId eq 202504" in url
    assert "CountryId eq 400" in url
    assert "FlowTypeId eq 4" in url
    assert "groupby((MonthId,CountryId,CommodityId)" in url
    assert "NetMass with sum as NetMass" in url
