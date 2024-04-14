import git
from git.exc import InvalidGitRepositoryError


def get_repo() -> git.Repo:
    return git.Repo(search_parent_directories=True)


def latest_commit(repo=None, file: str = None) -> str:
    if repo is None:
        try:
            repo = get_repo()
        except InvalidGitRepositoryError:
            return None
    if file is None:
        return repo.head.object.hexsha
    # this is probably implemented in gitpython
    log_out = repo.git.log(file)  # "commit 3d0efcda7eb0d4d803fe94c2107207a16a33e580\nAuthor: XYZ...""
    return log_out.split("\n")[0].split(" ")[1]
