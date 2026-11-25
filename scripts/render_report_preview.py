"""Render the report with the installed document skill, keeping preview/temp files in Project."""
import importlib.util,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
plugin_cache=Path.home()/'Library/Application Support/dev.nhaterik.AI-Account-Switcher/accounts/codex/053ece95-bbb8-4283-8b49-b2bba282e9c0/plugins/cache/openai-primary-runtime/documents'
renderers=list(plugin_cache.glob('*/skills/documents/render_docx.py'))
if not renderers:raise SystemExit('The managed document renderer is not installed.')
renderer=max(renderers,key=lambda p:p.stat().st_mtime)
spec=importlib.util.spec_from_file_location('document_renderer',renderer);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# env.sh has already placed TMPDIR inside Project; retain that choice for the bundled office process.
module._default_macos_tmpdir_for_soffice=lambda:None
sys.argv=[str(renderer),str(ROOT/'reports/bao_cao_do_an_hoc_may.docx'),'--output_dir',str(ROOT/'.cache/reliability-docx-preview'),'--emit_pdf']
module.main()
