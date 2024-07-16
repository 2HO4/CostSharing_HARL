from harl.common.base_logger import BaseLogger


class CostSharingLogger(BaseLogger):
    def get_task_name(self):
        return "cost_sharing"
