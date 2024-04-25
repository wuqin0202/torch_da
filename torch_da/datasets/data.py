from torch.utils.data import DataLoader

class CrossDomainLoader:
    def __init__(self, datasets, batch_size=1, shuffle=True, num_workers=4, mode='max'):
        self.loader_src = DataLoader(datasets[0], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
        self.loader_tgt = DataLoader(datasets[1], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
        if mode == 'max':
            self.num_iters = max(len(self.loader_src), len(self.loader_tgt))
        elif mode == 'min':
            self.num_iters = min(len(self.loader_src), len(self.loader_tgt))

    def __len__(self):
        return self.num_iters

    def __iter__(self):
        self.iter_src = iter(self.loader_src)
        self.iter_tar = iter(self.loader_tgt)
        self.cur_iter = 0
        return self

    def __next__(self):
        if self.cur_iter < self.num_iters:
            try:
                data_src, label_src = next(self.iter_src)
            except StopIteration:
                self.iter_src = iter(self.loader_src)
                data_src, label_src = next(self.iter_src)
            try:
                data_tar, label_tar = next(self.iter_tar)
            except StopIteration:
                self.iter_tar = iter(self.loader_tgt)
                data_tar, label_tar = next(self.iter_tar)
            self.cur_iter += 1
        else:
            self.cur_iter = 0
            raise StopIteration
        return data_src, label_src, data_tar, label_tar
