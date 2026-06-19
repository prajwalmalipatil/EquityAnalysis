import pytest
import responses
from datetime import datetime, timezone
from src.services.macro_intelligence.rbi_collector import RBICollector

@pytest.fixture
def collector():
    return RBICollector()

@responses.activate
def test_rbi_collector_pdf_extraction(collector):
    # Mock the official URL response containing a PDF link
    official_url = "https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=12345"
    mock_html = """
    <html>
      <body>
        <a href="https://rbidocs.rbi.org.in/rdocs/PressRelease/PDFs/PR12345.pdf">Download PDF</a>
        <a href="https://rbidocs.rbi.org.in/rdocs/content/pdfs/Accessibility20012026.pdf">Accessibility</a>
      </body>
    </html>
    """
    responses.add(responses.GET, official_url, body=mock_html, status=200)
    
    session = collector._get_retry_session()
    pdf_url = collector._extract_pdf_url(session, official_url)
    
    assert pdf_url == "https://rbidocs.rbi.org.in/rdocs/PressRelease/PDFs/PR12345.pdf"

@responses.activate
def test_rbi_collector_html_fallback(collector):
    # Mock the RSS feeds failing (status 500)
    for feed_url in collector.FEEDS.values():
        responses.add(responses.GET, feed_url, status=500)
        
    # Mock the HTML archive display pages
    pr_url = "https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx"
    mock_pr_html = """
    <html>
      <body>
        <table class="tablebg" width="100%">
          <tr><td class="tableheader" colspan="4" align="left"><b>Jun 19, 2026<b></td></tr>
          <tr>
            <td><a class='link2' href=BS_PressReleaseDisplay.aspx?prid=62970>Result of Repo Auction</a></td>
            <td nowrap colspan=3><a id='pdf1' target='_blank' href='https://rbidocs.rbi.org.in/rdocs/PressRelease/PDFs/PR62970.pdf'>PDF</a></td>
          </tr>
        </table>
      </body>
    </html>
    """
    responses.add(responses.GET, pr_url, body=mock_pr_html, status=200)
    
    # Mock Notifications page as well
    notif_url = "https://www.rbi.org.in/scripts/NotificationUser.aspx"
    mock_notif_html = """
    <html>
      <body>
        <table class="tablebg" width="100%">
          <tr><td class="tableheader" colspan="4" align="left"><b>Jun 18, 2026</b></td></tr>
          <tr>
            <td><a class='link2' href=NotificationUser.aspx?Id=13514&Mode=0>Interest Rate Directions</a></td>
            <td nowrap colspan="3"><a id='pdf2' target="_blank" href="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NT13514.pdf">PDF</a></td>
          </tr>
        </table>
      </body>
    </html>
    """
    responses.add(responses.GET, notif_url, body=mock_notif_html, status=200)
    
    last_dt = datetime(2026, 6, 17, tzinfo=timezone.utc)
    events = collector.fetch_since(last_dt.isoformat())
    
    assert len(events) == 2
    
    # Verify first event (Press Release)
    assert events[0].official_data.title == "Result of Repo Auction"
    assert events[0].official_data.category == "Press Releases"
    assert events[0].official_data.pdf_url == "https://rbidocs.rbi.org.in/rdocs/PressRelease/PDFs/PR62970.pdf"
    
    # Verify second event (Notification)
    assert events[1].official_data.title == "Interest Rate Directions"
    assert events[1].official_data.category == "Notifications"
    assert events[1].official_data.pdf_url == "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NT13514.pdf"
