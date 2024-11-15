from mongoengine import connect
from pycoshark.mongomodels import Commit, FileAction, File, Project, VCSSystem, Hunk, Issue
from pycoshark.utils import create_mongodb_uri_string
from bson import json_util

use_mongodb = True
credentials = {'db_user': '',
               'db_password': '',
               'db_hostname': 'localhost',
               'db_port': 27017,
               'db_authentication_database': '',
               'db_ssl_enabled': False}

database_name = 'smartshark_1_2'


def generate_hunk_labels():
    if use_mongodb:
        uri = create_mongodb_uri_string(**credentials)
        connect(database_name, host=uri, alias='default')

        completed = []
        # cache hunks locally to avoid timeouts
        tmp_hunks = [h for h in Hunk.objects(lines_manual__exists=True).only('id', 'lines_manual', 'file_action_id')]
        for h in tmp_hunks:
            if len(h.lines_manual) > 3:
                fa = FileAction.objects(id=h.file_action_id).get()
                file = File.objects(id=fa.file_id).get()
                commit = Commit.objects(id=fa.commit_id).only('revision_hash', 'fixed_issue_ids', 'vcs_system_id').get()
                vcs = VCSSystem.objects(id=commit.vcs_system_id).get()
                project = Project.objects(id=vcs.project_id).get()
                external_id = None
                num_fixed_bugs = 0
                for issue in Issue.objects(id__in=commit.fixed_issue_ids):
                    if issue.issue_type_verified is not None and issue.issue_type_verified.lower() == 'bug':
                        num_fixed_bugs += 1
                        external_id = issue.external_id
                if num_fixed_bugs == 1:
                    completed.append({'lines_manual': h.lines_manual,
                                      'file': file.path,
                                      'issue_id': issue.external_id,
                                      'revision_hash': commit.revision_hash,
                                      'hunk_id': h.id,
                                      'repository_url': vcs.url,
                                      'project': project.name})
                else:
                    pass  # this is just in case we start labeling commits that link to multiple bugs

        # store to disk
        with open('../data/hunk_labels.json', 'w') as file:
            file.write(json_util.dumps(completed))
    else:
        print("skipping (use_mongodb==False)")


def check_tangled_commits(data):
    commit_labels = {}
    for item in data:
        revision = item["revision_hash"]
        # Collecting all unique types for a revision
        types = set()
        for key in item["lines_manual"]:
            types.update(item["lines_manual"][key].keys())
        if revision in commit_labels:
            commit_labels[revision].update(types)
        else:
            commit_labels[revision] = types

    # Marking commits as tangled or not-tangled based on the number of unique types
    results = {revision: "tangled" if len(types) > 1 else "not-tangled" for revision, types in commit_labels.items()}
    print(len(results))
    return results


if __name__ == '__main__':
    # generate_hunk_labels()
    with open('../data/hunk_labels.json', 'r') as file:
        data = json_util.loads(file.read())
        results = check_tangled_commits(data)
        # save results to disk
        with open('../data/tangled_commits.json', 'w') as file:
            file.write(json_util.dumps(results))
