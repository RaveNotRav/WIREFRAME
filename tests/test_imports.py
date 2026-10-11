import importlib
import pkgutil

import backend


def test_all_backend_imports():
    def import_submodules(package, name):
        results = []
        for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
            if "migrations" in module_name:
                continue
            try:
                importlib.import_module(module_name)
                results.append((module_name, True, None))
            except Exception as e:
                results.append((module_name, False, str(e)))
        return results
        
    failures = []
    for mod, success, err in import_submodules(backend, 'backend'):
        if not success:
            failures.append(f"{mod}: {err}")
            
    assert not failures, f"Failed to import modules: {failures}"
