import { ResearchInvestigation } from '../types';

export function exportReportToPdf(investigation: ResearchInvestigation) {
  if (!investigation?.report) return;
  const r = investigation.report;
  const isInsufficient = investigation.status === 'insufficient_evidence';
  const isWarnings = investigation.status === 'completed_with_warnings';

  const esc = (s: string) =>
    (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const formatTimestamp = (dateStr?: string | Date) => {
    const d = dateStr ? new Date(dateStr) : new Date();
    return d.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      timeZoneName: 'short',
    });
  };

  const mdToHtml = (text: string) => {
    if (!text) return '';
    let out = esc(text);
    // Convert ### Headings
    out = out.replace(/^###\s+(.*?)$/gm, '<h3 style="font-size:10.5pt; font-weight:700; margin:14px 0 6px; color:#111;">$1</h3>');
    out = out.replace(/^##\s+(.*?)$/gm, '<h2 style="font-size:11.5pt; font-weight:700; margin:16px 0 6px; color:#111; border-bottom:1px solid #eee;">$1</h2>');
    out = out.replace(/^#\s+(.*?)$/gm, '<h1 style="font-size:13pt; font-weight:700; margin:18px 0 8px; color:#111;">$1</h1>');
    // Convert **bold** and *italic*
    out = out.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Convert inline code
    out = out.replace(/`([^`]+)`/g, '<code style="background:#f4f4f5; padding:2px 4px; border-radius:3px; font-family:monospace; font-size:9pt;">$1</code>');
    // Convert bullet points
    out = out.replace(/^[-*]\s+(.*?)$/gm, '<li style="margin-left:18px; margin-bottom:4px;">$1</li>');
    // Convert numbered lists
    out = out.replace(/^\d+\.\s+(.*?)$/gm, '<li style="margin-left:18px; margin-bottom:4px;">$1</li>');
    // Convert inline citations [1], [1, 2]
    out = out.replace(/\[(\d+(?:,\s*\d+)*)\]/g, '<sup style="color:#2563eb; font-weight:700; font-size:8pt; margin-left:1px;">[$1]</sup>');
    return out;
  };

  const comparisonTableHtml = r.comparisonTable?.length ? `
    <section>
      <h2>Performance Comparison</h2>
      <table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:8.5pt;">
        <thead>
          <tr style="background:#f4f4f5; text-align:left; border-bottom:1px solid #ddd;">
            <th style="padding:6px;">Method / Model</th>
            <th style="padding:6px;">Approach</th>
            <th style="padding:6px;">Dataset</th>
            <th style="padding:6px;">Quality Impact</th>
            <th style="padding:6px;">Comp. Ratio</th>
            <th style="padding:6px;">Speedup</th>
            <th style="padding:6px;">Memory</th>
            <th style="padding:6px;">Compute</th>
          </tr>
        </thead>
        <tbody>
          ${r.comparisonTable.map(row => `
            <tr style="border-bottom:1px solid #eee;">
              <td style="padding:6px; font-weight:600;">${esc(row.model)}</td>
              <td style="padding:6px; color:#555;">${esc(row.architectureType)}</td>
              <td style="padding:6px; font-family:monospace; font-size:8pt;">${esc(row.dataset)}</td>
              <td style="padding:6px; font-weight:600;">${esc(row.f1Score)}</td>
              <td style="padding:6px;">${esc(row.mapScore)}</td>
              <td style="padding:6px;">${esc(row.fpsThroughput)}</td>
              <td style="padding:6px; color:#555;">${esc(row.parametersM)}</td>
              <td style="padding:6px; color:#555;">${esc(row.gflops)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </section>
  ` : '';

  const findingsSections = r.findings.map(f => `
    <section>
      <h2>${esc(f.sectionTitle)}</h2>
      ${f.paragraphs.map(p => `
        <p>${mdToHtml(p.text)} ${p.citations?.map(c => `<span style="font-size:8pt; vertical-align:super; color:#2563eb; font-weight:bold;">[${c.badgeNumber}]</span>`).join(' ') || ''}</p>
      `).join('')}
    </section>`).join('');

  const limitationsList = r.limitations?.length
    ? `<section><h2>Limitations</h2><ul>${r.limitations.map(l => `<li>${esc(l)}</li>`).join('')}</ul></section>`
    : '';

  const referencesList = r.references?.length
    ? `<section class="references"><h2>References</h2><ol>${r.references.map(ref =>
        `<li><span class="ref-authors">${esc(ref.authors.join(', '))}</span> (${ref.publicationYear}). <em>${esc(ref.title)}</em>. ${esc(ref.journalConference)}.${ref.doi ? ` DOI: ${esc(ref.doi)}` : ''}</li>`
      ).join('')}</ol></section>`
    : '';

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>ResearchLens Report — ${esc(investigation.question.slice(0, 60))}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    
    @page {
      margin: 0;
      size: A4 portrait;
    }

    body {
      font-family: 'Inter', system-ui, sans-serif;
      font-size: 10.5pt;
      line-height: 1.65;
      color: #111;
      background: #fff;
      padding: 0;
      margin: 0;
    }

    .page-wrapper {
      box-sizing: border-box;
      max-width: 760px;
      margin: 0 auto;
      padding: 20mm 20mm 25mm;
    }

    .cover {
      padding: 0 0 24px 0;
      border-bottom: 2px solid #111;
      margin-bottom: 28px;
    }
    .cover .brand {
      font-size: 10pt;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #555;
      margin-bottom: 16px;
    }
    .cover h1 {
      font-size: 20pt;
      font-weight: 700;
      line-height: 1.25;
      margin-bottom: 18px;
      color: #111;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 24px;
      font-size: 9.5pt;
      color: #444;
    }
    .meta-grid .label { font-weight: 600; color: #111; }
    .stats-row {
      display: flex;
      gap: 12px;
      margin-top: 18px;
      flex-wrap: wrap;
    }
    .stat-pill {
      padding: 4px 12px;
      border: 1px solid #ddd;
      border-radius: 99px;
      font-size: 8.5pt;
      font-weight: 500;
      color: #333;
      background: #fafafa;
    }
    .content { padding: 0; }
    section { margin-bottom: 26px; page-break-inside: avoid; }
    h2 {
      font-size: 13pt;
      font-weight: 700;
      color: #111;
      margin-bottom: 10px;
      padding-bottom: 5px;
      border-bottom: 1px solid #eee;
    }
    p { margin-bottom: 10px; color: #222; }
    ul, ol { padding-left: 20px; color: #222; }
    li { margin-bottom: 5px; }
    .references ol { list-style-type: decimal; }
    .references li { font-size: 9.5pt; color: #333; margin-bottom: 6px; }
    .ref-authors { font-weight: 500; }
    .footer {
      margin-top: 36px;
      padding-top: 12px;
      border-top: 1px solid #eee;
      font-size: 8.5pt;
      color: #888;
      text-align: center;
    }
    @media print {
      body { background: white; }
      .page-wrapper {
        padding: 16mm 18mm 20mm;
        max-width: 100%;
      }
      .cover { padding-bottom: 20px; }
      h2 { page-break-after: avoid; }
      section { page-break-inside: avoid; }
      .references { page-break-before: always; }
    }
  </style>
</head>
<body>
  <div class="page-wrapper">
    <div class="cover">
      <div class="brand">ResearchLens · AI Research Report</div>
      <h1>${esc(investigation.question)}</h1>
      <div class="meta-grid">
        <div><span class="label">Research Depth</span></div><div>${esc(investigation.depth)}</div>
        <div><span class="label">Date Generated</span></div><div>${formatTimestamp(investigation.updatedAt)}</div>
        <div><span class="label">Sources</span></div><div>${investigation.sources.map(esc).join(', ')}</div>
        <div><span class="label">Status</span></div><div style="font-weight:600; color:${isInsufficient ? '#ea580c' : (isWarnings ? '#d97706' : '#16a34a')}">${isInsufficient ? 'Insufficient evidence' : (isWarnings ? 'Completed with warnings' : 'Completed')}</div>
      </div>
      <div class="stats-row">
        <span class="stat-pill">📄 ${investigation.papersAnalyzed} Papers Analyzed</span>
        <span class="stat-pill">✅ ${investigation.verifiedClaims} Verified Claims</span>
        <span class="stat-pill">🔍 ${investigation.evidenceItems} Evidence Items</span>
        <span class="stat-pill">📊 Citation Integrity: ${investigation.citationCoverage != null ? `${investigation.citationCoverage}%` : '100%'}</span>
        <span class="stat-pill">⚠️ ${investigation.potentialConflicts ?? 0} Conflicts</span>
      </div>
    </div>

    <div class="content">
      ${isWarnings ? `
      <section style="border:1px solid #f59e0b; border-radius:6px; padding:12px 16px; background:#fffbeb; margin-bottom:20px;">
        <h3 style="color:#b45309; font-size:10pt; font-weight:700; margin-bottom:4px;">⚠️ Completed with Warnings</h3>
        <p style="color:#92400e; font-size:9pt; margin:0;">${esc(investigation.failure_reason || (investigation.debug?.status_reasons || []).join('; ') || 'Anchor or citation integrity constraints detected.')}</p>
      </section>
      ` : ''}
      ${isInsufficient ? `
      <section style="border:2px solid #ea580c; border-radius:8px; padding:20px 24px; background:#fff7ed; margin-bottom:28px;">
        <h2 style="color:#ea580c; border-bottom:1px solid #fdba74;">&#9888; Insufficient Evidence Found</h2>
        <p style="white-space:pre-line; color:#9a3412; font-size:10pt; line-height:1.8;">${esc(r.executiveSummary)}</p>
      </section>
      <section>
        <h2>Retrieved Sources</h2>
        ${referencesList || '<p style="color:#888;">No papers were indexed.</p>'}
      </section>
      ` : `
      <section>
        <h2>Executive Summary</h2>
        <div style="line-height:1.7;">${mdToHtml(r.executiveSummary)}</div>
      </section>

      <section>
        <h2>Research Methodology &amp; Protocol</h2>
        <div style="line-height:1.7;">${mdToHtml(r.methodology)}</div>
      </section>

      ${findingsSections}

      ${comparisonTableHtml}

      ${r.computationalRequirements ? `<section><h2>Computational Requirements</h2><div style="line-height:1.7;">${mdToHtml(r.computationalRequirements)}</div></section>` : ''}
      ${r.contradictoryEvidence ? `<section><h2>Contradictory Evidence</h2><div style="line-height:1.7;">${mdToHtml(r.contradictoryEvidence)}</div></section>` : ''}
      ${limitationsList}

      <section>
        <h2>Conclusion &amp; Decision Framework</h2>
        <div style="line-height:1.7;">${mdToHtml(r.conclusion)}</div>
      </section>

      ${referencesList}
      `}

      <div class="footer">
        Generated by ResearchLens · ${formatTimestamp()} · For academic reference only
      </div>
    </div>
  </div>

  <script>
    window.onload = function () {
      window.print();
    };
  </script>
</body>
</html>`;

  const printWindow = window.open('', '_blank', 'width=900,height=700');
  if (printWindow) {
    printWindow.document.write(html);
    printWindow.document.close();
  }
}
